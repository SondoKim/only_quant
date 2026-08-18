# -*- coding: utf-8 -*-
"""금리 북 정비 A/B (2026-08-14) — regime gate 존폐 + value/carry 재판정.

동기 (2026-08-14 재검증, memory rates-review-2026-08):
  ① regime gate(Hurst): 2016+ 진단에서 ON 1.03 vs OFF 1.04 (원시 선물 귀속)
     — 효과가 0 이면 죽은 복잡성이므로 제거한다. 리버전 OFF·스톱 수정 이후
     정식 재검증된 적 없음.
  ② value/carry: 단독 SR 이 각각 -0.06 / -0.61 (net) 인데 leave-one-out 은
     +0.08 / +0.13 기여 — 분산 기여가 실재인지 노이즈인지 양 표본·전 열에서
     재판정한다.
  ③ (진단) carry 반전: "단독 -0.61 이면 뒤집으면 +0.61?" 에 대한 실측 답.
     채택 후보가 아님 — 같은 표본 부호 채굴은 선택편의의 정의 그 자체.

⚠ sleeve_weights 는 금리/FX 북이 공유한다 (xs_neutralize 와 달리 FX 오버라이드
없음). 단 vol_target_mode=separate 라 금리 북 포지션은 FX 와 완전 독립 —
금리 SR 열은 오염되지 않는다. FX SR 열은 참고용으로만 표시 (value/carry 를
실제로 제거하려면 sleeve_weights_fx 오버라이드를 먼저 만들어야 함).

프로토콜: 2016+/2012+, H1/H2, T+2, 시간대정직, net of costs, yield-implied
귀속. 격자 사전등록·사후 세분화 재튜닝 금지.

판정 기준 (사전등록):
  - regime gate 제거: 양 표본 · 전 열에서 |ΔSR| ≤ 0.03 (동치) 또는 개선이면
    제거 채택 (단순화 우선). 한 열이라도 유의미 악화(-0.05 초과)면 유지.
  - value/carry 제거: 양 표본 · 전 열 전반 개선일 때만 제거 (기존 채택 룰).

Usage: python scripts/test_cleanup_ab.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader                            # noqa: E402
from src.data.preprocessor import DataPreprocessor                # noqa: E402
from src.sleeves.sleeve_engine import SleeveEngine, TRADING_DAYS  # noqa: E402
from scripts.run_sleeve_backtest import (                         # noqa: E402
    load_sleeve_config, perf_stats, cost_bps_for, DEFAULT_COSTS_BPS)
from scripts.test_curve_blocks import yield_implied_returns       # noqa: E402

EARLY = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
         'YM1 Comdty', 'XM1 Comdty']

# (이름, regime_enabled, sleeve_weights 오버라이드)
CASES = [
    ('A. 현행',                 True,  {}),
    ('B. regime gate OFF',      False, {}),
    ('C. value 제거',           True,  {'value': 0.0}),
    ('D. carry 제거',           True,  {'carry': 0.0}),
    ('E. value+carry 제거',     True,  {'value': 0.0, 'carry': 0.0}),
    ('F. B+C+D (모멘텀만)',     False, {'value': 0.0, 'carry': 0.0}),
    # ── 진단 전용 (채택 불가) ──
    ('G. carry 단독',           True,  {'trend': 0.0, 'value': 0.0,
                                        'policy': 0.0, 'carry': 1.0}),
    ('H. carry 단독 반전',      True,  {'trend': 0.0, 'value': 0.0,
                                        'policy': 0.0, 'carry': -1.0}),
]


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}

    def run(regime_on, w_ov, start):
        cfg = {**BASE}
        cfg['regime_gate'] = {**(BASE.get('regime_gate') or {}),
                              'enabled': regime_on}
        cfg['sleeve_weights'] = {**(BASE.get('sleeve_weights') or {}), **w_ov}
        px = PX[PX.index >= pd.to_datetime(start)]
        e = SleeveEngine(px, config=cfg, yields=YL)
        R = e.rates_assets
        F = e.fx_assets
        pos = e.finalize_positions(e.compute_target_positions())

        dirr = e.dir_returns[R + F].reindex(pos.index).fillna(0.0)
        rets = yield_implied_returns(e, dirr)
        crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R + F})
        turn = pos[R + F].diff().abs().fillna(0.0)

        def pnl(cols, lag_map):
            held = pd.DataFrame({a: pos[a].shift(lag_map.get(a, 1))
                                 for a in cols}).fillna(0.0)
            return (held * rets[cols]
                    - turn[cols].mul(crate[cols], axis=1)).sum(axis=1).dropna()

        s1 = pnl(R, {a: 1 for a in R})
        s2 = pnl(R, {a: 2 for a in R})
        stz = pnl(R, {a: (2 if a in EARLY else 1) for a in R})
        fx1 = pnl(F, {a: 1 for a in F})
        st = perf_stats(s1)
        mid = len(s1) // 2
        return {
            'sr': st['sharpe'], 'ret': st['ret'], 'maxdd': st['maxdd'],
            'h1': perf_stats(s1.iloc[:mid])['sharpe'],
            'h2': perf_stats(s1.iloc[mid:])['sharpe'],
            't2': perf_stats(s2)['sharpe'], 'tz': perf_stats(stz)['sharpe'],
            'turn': float(turn[R].sum(axis=1).mean() * TRADING_DAYS),
            'fx': perf_stats(fx1)['sharpe'],
        }

    for start in ['2016-01-01', '2012-01-01']:
        print("\n" + "=" * 110)
        print(f"  금리 북 정비 A/B — {start[:4]}+  "
              f"(G/H 는 진단 전용 — 채택 대상 아님)")
        print("=" * 110)
        print(f"{'구성':<22}{'SR':>7}{'H1':>6}{'H2':>6}{'T+2':>6}"
              f"{'시간대정직':>10}{'AnnRet':>8}{'MaxDD':>8}{'회전':>7}{'FX SR':>7}")
        for name, regime_on, w_ov in CASES:
            r = run(regime_on, w_ov, start)
            print(f"{name:<22}{r['sr']:>7.2f}{r['h1']:>6.2f}{r['h2']:>6.2f}"
                  f"{r['t2']:>6.2f}{r['tz']:>10.2f}{r['ret']:>8.2%}"
                  f"{r['maxdd']:>8.1%}{r['turn']:>6.0f}x{r['fx']:>7.2f}")

    print("\n  * FX SR 열: sleeve_weights 가 북 공유라 C~H 에서 FX 도 변한다 —")
    print("    참고용. vol_target_mode=separate 라 금리 열은 FX 와 완전 독립.")
    print("  * H(carry 반전) 는 G 의 부호 반전이 아니다: 비용은 방향과 무관하게")
    print("    빠지고, 북스톱·볼타겟이 자기 손익 경로에 반응하는 비선형 오버레이라")
    print("    반전 북은 다른 스톱 패턴을 탄다. 무엇보다 같은 표본 부호 채굴은 금지.")


if __name__ == '__main__':
    main()
