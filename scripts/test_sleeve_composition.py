# -*- coding: utf-8 -*-
"""금리 슬리브 구성 최적성 전수 검증 (2026-08-31).

질문: trend/value/carry/policy 4-슬리브 균등(각 1.0)이 현재 모델에서 최적인가?

기존 근거 (부분 측정만 존재):
  - 2026-08-14 leave-one-out: value 빼면 1.06, carry 빼면 0.97 (베이스 1.16,
    2016+) — 하나씩 빼면 악화. 단독 SR: trend 0.69, policy 0.91, value −0.06,
    carry −0.61. trend↔policy 상관 +0.84 (실질 독립 팩터는 방향성 모멘텀 하나).
  - 2026-06-12: value 가중 0→0.5→1.0 단조 개선으로 1.0 채택.
  하지만 15개 부분집합 '전수'와 가중 민감도 격자는 잰 적 없다.

측정 (사전등록):
  ① 부분집합 15종 전체 — 각 슬리브 0/1 (포함=1.0, 제외=0.0).
  ② 가중 민감도 — 한 슬리브만 0.5 또는 1.5, 나머지 1.0 (one-at-a-time 8종).
  ③ 모멘텀 블록 반감 — trend 0.5 + policy 0.5 (상관 0.84 중복 블록의 상대
     비중을 value/carry 대비 절반으로; 볼타겟이 전체 스케일은 흡수하므로
     상대 비중만 유효).

프로토콜: 엔진 2010+ 전기간 1회/구성, 손익 윈도잉. yield-implied 귀속,
net of costs, T+2·시간대정직 열 포함. 개발 2012-2021 / 홀드아웃 2022+.

판정 게이트 (사전등록, 기존 규율 동일):
  - 개발 전 열(SR/H1/H2/T+2/시간대정직) 개선(허용오차 −0.03)일 때만 홀드아웃
    열람 → 홀드아웃 ΔSR ≥ −0.03 이면 채택 후보.
  - 개발 게이트 탈락 구성의 홀드아웃은 출력하지 않는다 (홀드아웃 재사용 최소화).
  - ⚠ 이 홀드아웃은 이미 여러 라운드에서 열람됐다 — 증거력 저하 상태이므로
    아슬아슬한 통과는 채택하지 않는다.

⚠ sleeve_weights 는 금리/FX 북 공유 (FX 오버라이드 없음). vol_target_mode=
separate 라 금리 열은 오염되지 않는다. 채택 시 FX 영향은 별도 확인 필요.

Usage: python scripts/test_sleeve_composition.py
"""
import itertools
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

SLEEVES = ['trend', 'value', 'carry', 'policy']
DEV = ('2012-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)
EARLY = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
         'YM1 Comdty', 'XM1 Comdty']
TOL = 0.03


def window(s, a, b):
    s = s[s.index >= pd.to_datetime(a)]
    if b:
        s = s[s.index <= pd.to_datetime(b)]
    return s.dropna()


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}

    def run(w_ov):
        cfg = {**BASE}
        cfg['sleeve_weights'] = {**(BASE.get('sleeve_weights') or {}), **w_ov}
        e = SleeveEngine(PX, config=cfg, yields=YL)
        R = e.rates_assets
        pos = e.finalize_positions(e.compute_target_positions())
        dirr = e.dir_returns[R + e.fx_assets].reindex(pos.index).fillna(0.0)
        rets = yield_implied_returns(e, dirr)
        crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})
        turn = pos[R].diff().abs().fillna(0.0)

        def pnl(lag_map):
            held = pd.DataFrame({a: pos[a].shift(lag_map.get(a, 1))
                                 for a in R}).fillna(0.0)
            return (held * rets[R]
                    - turn.mul(crate, axis=1)).sum(axis=1).dropna()

        s1 = pnl({a: 1 for a in R})
        s2 = pnl({a: 2 for a in R})
        stz = pnl({a: (2 if a in EARLY else 1) for a in R})
        d1 = window(s1, *DEV)
        mid = len(d1) // 2
        return {
            'dev': perf_stats(d1)['sharpe'],
            'h1': perf_stats(d1.iloc[:mid])['sharpe'],
            'h2': perf_stats(d1.iloc[mid:])['sharpe'],
            't2': perf_stats(window(s2, *DEV))['sharpe'],
            'tz': perf_stats(window(stz, *DEV))['sharpe'],
            's16': perf_stats(window(s1, '2016-01-01', None))['sharpe'],
            'hold': perf_stats(window(s1, *HOLD))['sharpe'],
            'turn': float(window(turn.sum(axis=1), *DEV).mean() * TRADING_DAYS),
        }

    GATE_COLS = ['dev', 'h1', 'h2', 't2', 'tz']

    def row(name, r, base, show_hold):
        hold = f"{r['hold']:>6.2f}" if show_hold else '     —'
        print(f"{name:<26}{r['dev']:>7.2f}{r['h1']:>6.2f}{r['h2']:>6.2f}"
              f"{r['t2']:>6.2f}{r['tz']:>6.2f}{r['s16']:>8.2f}{hold}"
              f"{r['turn']:>7.0f}x")

    def dev_pass(r, base):
        return all(r[c] >= base[c] - TOL for c in GATE_COLS)

    hdr = (f"{'구성':<26}{'devSR':>7}{'H1':>6}{'H2':>6}{'T+2':>6}{'정직':>6}"
           f"{'2016+':>8}{'hold':>6}{'회전':>7}")

    # ── ① 부분집합 15종 ──────────────────────────────────────────────
    print('=' * 84)
    print('  ① 슬리브 부분집합 전수 (포함=1.0 / 제외=0.0) — 개발 2012-2021 판정')
    print('=' * 84)
    print(hdr)
    base = run({})
    row('베이스 (현행 4종)', base, base, True)
    passers = []
    for k in (3, 2, 1):
        for combo in itertools.combinations(SLEEVES, k):
            ov = {s: (1.0 if s in combo else 0.0) for s in SLEEVES}
            r = run(ov)
            ok = dev_pass(r, base)
            name = '+'.join(combo)
            row(('✓ ' if ok else '  ') + name, r, base, ok)
            if ok:
                passers.append((name, r))

    # ── ② 가중 민감도 + ③ 모멘텀 블록 반감 ──────────────────────────
    print()
    print('=' * 84)
    print('  ② 가중 민감도 (one-at-a-time) + ③ trend 0.5+policy 0.5')
    print('=' * 84)
    print(hdr)
    row('베이스 (전 슬리브 1.0)', base, base, True)
    for s in SLEEVES:
        for w in (0.5, 1.5):
            r = run({s: w})
            ok = dev_pass(r, base)
            name = f'{s} {w}'
            row(('✓ ' if ok else '  ') + name, r, base, ok)
            if ok:
                passers.append((name, r))
    r = run({'trend': 0.5, 'policy': 0.5})
    ok = dev_pass(r, base)
    row(('✓ ' if ok else '  ') + 'trend 0.5 + policy 0.5', r, base, ok)
    if ok:
        passers.append(('trend 0.5 + policy 0.5', r))

    # ── 판정 ─────────────────────────────────────────────────────────
    print()
    print('  게이트: 개발 전 열 ≥ 베이스 −0.03 → 홀드아웃 열람 → hold ΔSR ≥ −0.03')
    if not passers:
        print('  → 개발 게이트 통과 구성 없음: 현행 4-슬리브 균등이 국소 최적으로 확인.')
    else:
        for name, r in passers:
            verdict = ('홀드아웃 통과' if r['hold'] >= base['hold'] - TOL
                       else '홀드아웃 탈락')
            print(f"  → {name}: dev {r['dev']:.2f} (베이스 {base['dev']:.2f}), "
                  f"hold {r['hold']:.2f} (베이스 {base['hold']:.2f}) — {verdict}")
        print('  ⚠ 홀드아웃 재사용 상태 — 아슬아슬한 통과는 채택하지 않는다.')


if __name__ == '__main__':
    main()
