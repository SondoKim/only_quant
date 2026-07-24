# -*- coding: utf-8 -*-
"""금리 trend/policy 슬리브의 횡단면(xs) 부분 중립화 A/B (2026-07-23).

동기: 현행 북에서 trend/policy 는 순수 시계열(방향성, xs_neutralize=0)이고
상대가치 성분은 value/carry 에만 있다. 국가 간 상대 금리 모멘텀(예: BOK 가
Fed 보다 빨리 완화 → KTB 롱 오버웨이트/UST 언더)은 별도 문서화된 프리미엄
이므로, trend/policy 에 부분 중립화(0.25/0.5/1.0)를 얹어 상대 베팅 성분을
추가하는 것을 검증한다. 시그널 유니버스 8종이 횡단면 기준선을 만들고
포지션은 韓美 4종만 — value/carry 와 같은 '반쪽 헤지' 구조다.

FX 북은 xs_neutralize_fx 로 현행값에 고정 — 이 실험은 금리 북만 바꾼다
(포트폴리오 볼타겟 결합을 통한 2차 효과만 남음; FX SR 열로 감시).

프로토콜: 2016+/2012+, H1/H2, T+2, 시간대정직, net of costs. 격자는 사전등록
(0.25/0.5/1.0 단독·결합)이며 사후 세분화 재튜닝 금지.

Usage: python scripts/test_xs_neutralize.py
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

FX_PIN = {'trend': 0.0, 'value': 1.0, 'carry': 1.0}   # FX 북 현행 고정

CASES = [
    ('A. 현행 t0/p0',       0.00, 0.00),
    ('B. trend .25',        0.25, 0.00),
    ('C. trend .5',         0.50, 0.00),
    ('D. trend 1.0',        1.00, 0.00),
    ('E. policy .25',       0.00, 0.25),
    ('F. policy .5',        0.00, 0.50),
    ('G. policy 1.0',       0.00, 1.00),
    ('H. t.25 + p.25',      0.25, 0.25),
    ('I. t.5 + p.5',        0.50, 0.50),
]


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}

    def run(t_xs, p_xs, start):
        cfg = {**BASE}
        cfg['xs_neutralize'] = {**(BASE.get('xs_neutralize') or {}),
                                'trend': t_xs, 'policy': p_xs}
        cfg['xs_neutralize_fx'] = FX_PIN
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
        print("\n" + "=" * 108)
        print(f"  금리 trend/policy xs 중립화 A/B — {start[:4]}+  "
              f"(FX 는 현행 고정; 시간대정직 = 조기마감 T+2·미국 T+1)")
        print("=" * 108)
        print(f"{'구성':<20}{'SR':>7}{'H1':>6}{'H2':>6}{'T+2':>6}"
              f"{'시간대정직':>10}{'AnnRet':>8}{'MaxDD':>8}{'회전':>7}{'FX SR':>7}")
        for name, t_xs, p_xs in CASES:
            r = run(t_xs, p_xs, start)
            print(f"{name:<20}{r['sr']:>7.2f}{r['h1']:>6.2f}{r['h2']:>6.2f}"
                  f"{r['t2']:>6.2f}{r['tz']:>10.2f}{r['ret']:>8.2%}"
                  f"{r['maxdd']:>8.1%}{r['turn']:>6.0f}x{r['fx']:>7.2f}")

    print("\n  * 상대 베팅의 반대편 다리(英日豪)는 signal_only 라 포지션이 0 —")
    print("    반쪽 헤지 구조. FX SR 열은 볼타겟 결합 2차 효과 감시용 (변화 없어야 정상).")
    print("  * 판정: 2016+·2012+ 모두 H1·H2·T+2·시간대정직 전반 개선 시에만 채택.")


if __name__ == '__main__':
    main()
