# -*- coding: utf-8 -*-
"""금리 북 방향성 A/B — 방향성 슬리브를 중립화하면 무엇을 잃고 얻나.

배경 (2026-07-22): 리버전 서브북(가짜, 비동시 종가 아티팩트)을 폐기하자
방향성 북이 자리의 100%를 차지하게 됐다 — |net|/gross 중앙값 0.38 → 0.91.
방향은 xs_neutralize=0 인 trend·policy 에서 나온다 (value·carry 는 횡단면 합
정확히 0). 방향성을 줄이고 싶다면 손잡이는 policy 의 중립화 강도다.

과최적화 방지: 전·후반 분할 + T+2 지연 SR + 2012/2016 두 창을 항상 병기.
방향성 감소는 그 자체로 목적이 아니라 리스크 선호의 문제이므로, SR 뿐 아니라
|net|/gross 와 MaxDD 를 같이 보고 판단할 것.

Usage: python scripts/test_directionality.py
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

# 미국(시카고)보다 먼저 마감 — 당일 미국 종가를 볼 수 없다 (시간대 정직 테스트용)
EARLY = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
         'YM1 Comdty', 'XM1 Comdty']

BASE_XN = {'trend': 0.0, 'value': 1.0, 'carry': 1.0}

CASES = [
    ('A. 현행 (policy 방향성)',      {'xs_neutralize': {**BASE_XN, 'policy': 0.0}}),
    ('B. policy 절반 중립',          {'xs_neutralize': {**BASE_XN, 'policy': 0.5}}),
    ('C. policy 완전 중립',          {'xs_neutralize': {**BASE_XN, 'policy': 1.0}}),
    ('D. policy 제거 (가중 0)',      {'sleeve_weights_policy': 0.0}),
    ('E. trend 도 절반 중립',        {'xs_neutralize': {**BASE_XN, 'trend': 0.5,
                                                       'policy': 0.0}}),
    ('F. 둘 다 완전 중립',           {'xs_neutralize': {'trend': 1.0, 'value': 1.0,
                                                       'carry': 1.0, 'policy': 1.0}}),
]


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}

    def run(ov, start):
        cfg = {**BASE}
        ov = dict(ov)
        pw = ov.pop('sleeve_weights_policy', None)
        cfg.update(ov)
        if pw is not None:
            cfg['sleeve_weights'] = {**cfg['sleeve_weights'], 'policy': pw}
        px = PX[PX.index >= pd.to_datetime(start)]
        e = SleeveEngine(px, config=cfg, yields=YL)
        R = e.rates_assets
        pos = e.finalize_positions(e.compute_target_positions())

        dirr = e.dir_returns[R].reindex(pos.index).fillna(0.0)
        rets = dirr.copy()
        for a in R:
            yt = e.tradeable_yield_map.get(a)
            if yt is None or e.yields is None or yt not in e.yields.columns:
                continue
            dy = e.yields[yt].reindex(dirr.index).diff() * 100.0
            b = (dirr[a].rolling(250, min_periods=60).cov(dy)
                 / dy.rolling(250, min_periods=60).var()).shift(1)
            v = b * dy
            rets[a] = v.where(v.notna(), dirr[a])
        crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})
        turn = pos[R].diff().abs().fillna(0.0)

        def pnl(lag_map):
            held = pd.DataFrame({a: pos[a].shift(lag_map.get(a, 1))
                                 for a in R}).fillna(0.0)
            return (held * rets[R] - turn.mul(crate, axis=1)).sum(axis=1).dropna()

        s1 = pnl({a: 1 for a in R})
        s_tz = pnl({a: (2 if a in EARLY else 1) for a in R})
        st = perf_stats(s1)
        mid = len(s1) // 2
        net, gro = pos[R].sum(axis=1), pos[R].abs().sum(axis=1)
        live = gro > 1e-9
        return {
            'sr': st['sharpe'], 'ret': st['ret'], 'maxdd': st['maxdd'],
            'h1': perf_stats(s1.iloc[:mid])['sharpe'],
            'h2': perf_stats(s1.iloc[mid:])['sharpe'],
            'tz': perf_stats(s_tz)['sharpe'],
            'dir': float((net[live].abs() / gro[live]).median()),
            'turn': float(turn.sum(axis=1).mean() * TRADING_DAYS),
        }

    for start in ['2016-01-01', '2012-01-01']:
        print("\n" + "=" * 100)
        print(f"  금리 북 방향성 A/B — {start[:4]}+  "
              f"(방향도 = |net|/gross 중앙값, 1.0=완전 한방향 / 0=완전 중립)")
        print("=" * 100)
        print(f"{'구성':<26}{'SR':>6}{'H1':>6}{'H2':>6}{'T+2':>6}"
              f"{'AnnRet':>8}{'MaxDD':>8}{'방향도':>7}{'회전':>7}")
        for name, ov in CASES:
            r = run(ov, start)
            print(f"{name:<26}{r['sr']:>6.2f}{r['h1']:>6.2f}{r['h2']:>6.2f}"
                  f"{r['tz']:>6.2f}{r['ret']:>8.2%}{r['maxdd']:>8.1%}"
                  f"{r['dir']:>7.2f}{r['turn']:>6.0f}x")

    print("\n  * T+2 = 하루 늦게 집행. 본 SR 과 크게 벌어지면 시차 의존 의심.")
    print("  * 방향성 감소는 SR 극대화가 아니라 리스크 선호의 문제 — "
          "SR 손실 대비 방향도·MaxDD 개선폭으로 판단할 것.")


if __name__ == '__main__':
    main()
