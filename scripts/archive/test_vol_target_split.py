# -*- coding: utf-8 -*-
"""포트폴리오 볼타겟 분리 A/B (2026-07-24).

동기(운용 요청): 현행 `_apply_portfolio_vol_target` 는 금리 4종 + FX 5종을 한
포트폴리오로 묶어 하나의 target_port_vol 에 맞춘다. 그래서 FX 북 변동성이
튀면 금리 포지션이 같이 줄어든다 — 두 북을 별도 운용/별도 한도로 굴리려면
이 교차 의존을 끊어야 한다. `vol_target_mode: separate` 로 각 북이 자기
타겟을 독립 타게팅하도록 바꾸고 그 영향을 잰다.

이 실험은 '성과 개선'이 목적이 아니라 '운용 스킴 변경'이다. 판정 기준도
다르다 — SR 이 유지되는지(개선 요구 아님)와, 총위험이 얼마나 커지는지를
정량화해 타겟 재설정 근거를 만드는 것이 목적.

⚠ 분리하면 총위험이 커진다: 상관 ρ 인 두 북이 각각 v 를 타게팅하면 합산
변동성 ≈ v·√(2(1+ρ)). 그래서 케이스에 '합산 위험 보존' 변형을 함께 넣었다.

프로토콜: 2016+/2012+, H1/H2, T+2, 시간대정직, net of costs, 금리는
yield-implied PnL (롤 점프 제거).

Usage: python scripts/test_vol_target_split.py
"""
import sys
from pathlib import Path

import numpy as np
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

# 사전등록 케이스 (사후 세분화 재튜닝 금지)
CASES = [
    ('A. 현행 결합 10%',        'combined', 0.10, 0.10),
    ('B. 분리 각 10%',          'separate', 0.10, 0.10),
    ('C. 분리 각 7.1%(합산보존)', 'separate', 0.071, 0.071),
]


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}

    def run(mode, tv_r, tv_f, start):
        cfg = {**BASE, 'vol_target_mode': mode,
               'target_port_vol_rates': tv_r, 'target_port_vol_fx': tv_f}
        px = PX[PX.index >= pd.to_datetime(start)]
        e = SleeveEngine(px, config=cfg, yields=YL)
        R, F = e.rates_assets, e.fx_assets
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
        both = (s1 + fx1).dropna()
        st, stf, stb = perf_stats(s1), perf_stats(fx1), perf_stats(both)
        mid = len(s1) // 2
        return {
            'sr': st['sharpe'], 'vol': s1.std() * np.sqrt(TRADING_DAYS),
            'maxdd': st['maxdd'],
            'h1': perf_stats(s1.iloc[:mid])['sharpe'],
            'h2': perf_stats(s1.iloc[mid:])['sharpe'],
            't2': perf_stats(s2)['sharpe'], 'tz': perf_stats(stz)['sharpe'],
            'turn': float(turn[R].sum(axis=1).mean() * TRADING_DAYS),
            'fxsr': stf['sharpe'], 'fxvol': fx1.std() * np.sqrt(TRADING_DAYS),
            'bsr': stb['sharpe'], 'bvol': both.std() * np.sqrt(TRADING_DAYS),
            'bdd': stb['maxdd'],
            'corr': float(s1.corr(fx1)),
        }

    for start in ['2016-01-01', '2012-01-01']:
        print("\n" + "=" * 114)
        print(f"  볼타겟 분리 A/B — {start[:4]}+   "
              f"(금리 북은 rates_exposure_scale 0.5 적용 후 실현변동성)")
        print("=" * 114)
        print(f"{'구성':<24}{'금리SR':>7}{'금리vol':>8}{'금리DD':>8}"
              f"{'H1':>6}{'H2':>6}{'T+2':>6}{'시간대':>7}{'회전':>7}"
              f"{'FX SR':>7}{'FXvol':>7}{'합SR':>6}{'합vol':>7}{'합DD':>7}{'ρ':>6}")
        for name, mode, tvr, tvf in CASES:
            r = run(mode, tvr, tvf, start)
            print(f"{name:<24}{r['sr']:>7.2f}{r['vol']:>8.1%}{r['maxdd']:>8.1%}"
                  f"{r['h1']:>6.2f}{r['h2']:>6.2f}{r['t2']:>6.2f}{r['tz']:>7.2f}"
                  f"{r['turn']:>6.0f}x{r['fxsr']:>7.2f}{r['fxvol']:>7.1%}"
                  f"{r['bsr']:>6.2f}{r['bvol']:>7.1%}{r['bdd']:>7.1%}{r['corr']:>6.2f}")

    print("\n  * 판정 기준: 분리는 '운용 스킴 변경'이므로 SR 개선을 요구하지 않는다.")
    print("    확인할 것 — ① 금리 SR 이 유지되는가 ② 합산 변동성이 얼마나 커지는가")
    print("    ③ 금리 북이 FX 와 독립적으로 움직이는가(교차 의존 제거 확인).")


if __name__ == '__main__':
    main()
