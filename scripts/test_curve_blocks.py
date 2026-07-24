# -*- coding: utf-8 -*-
"""커브 트레이드 + 블록 리스크 예산 A/B (2026-07-23).

동기: 현행 금리 북은 ρ(TU,TY)=0.78, ρ(KE,KAA)=0.87 로 실효 독립 베팅이
'미국 듀레이션 + 한국 듀레이션' 2개뿐 ([[sleeve-engine]] 경고). 새 계약 없이
같은 韓美 4개 계약으로 베팅 수를 늘리는 두 장치를 검증한다:

  · curve_trades — US 2s10s / KR 3s10s vol-중립 스티프너를 합성 자산으로 추가.
    슬리브 = trend/value(합성 가격) + carry(−z(slope)) + policy(완화→불스팁),
    가중 사전등록 균등. 동일 국가·동시 마감 페어라 비동시 종가 아티팩트 없음.
  · risk_blocks — 국가 블록 내 평균 롤링 상관으로 블록을 '한 개 베팅' 리스크로
    정규화 (equal risk per bet, not per asset).

시차 감사 병기 (기존 프로토콜): T+2 와 시간대 정직(조기마감 T+2·미국 T+1) SR.
커브는 레그 공간(실계약 포지션)에서 평가 → 비용은 두 다리 회전에 실제로 부과.
커브 북 단독 진단은 합성 공간(explode_curves=False)에서 별도 산출하고,
두-다리 비용을 (1+H)·|Δp| 로 부과 + 0.5/1/2bp 비용 스윕.

Usage: python scripts/test_curve_blocks.py
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

EARLY = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
         'YM1 Comdty', 'XM1 Comdty']   # 미국보다 먼저 마감 (조기마감 T+2 대상)

CASES = [
    ('A. 현행',                {}),
    ('B. +블록예산',           {'risk_blocks': {'enabled': True}}),
    ('C. +커브',               {'curve_trades': {'enabled': True}}),
    ('D. +커브+블록예산',      {'curve_trades': {'enabled': True},
                                'risk_blocks': {'enabled': True}}),
]


def merged(base, ov):
    cfg = {**base}
    for k, v in ov.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def yield_implied_returns(e, dirr):
    """기존 프로토콜과 동일 — 금리 자산 손익을 '캐시금리 변동 × 회귀 베타'로
    환산 (연속선물 롤 점프 제거). 베타 워밍업/결측 구간은 선물수익률 폴백."""
    rets = dirr.copy()
    for a in [c for c in dirr.columns if 'Comdty' in c]:
        yt = e.tradeable_yield_map.get(a)
        if yt is None or e.yields is None or yt not in e.yields.columns:
            continue
        dy = e.yields[yt].reindex(dirr.index).diff() * 100.0
        b = (dirr[a].rolling(250, min_periods=60).cov(dy)
             / dy.rolling(250, min_periods=60).var()).shift(1)
        v = b * dy
        rets[a] = v.where(v.notna(), dirr[a])
    return rets


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}

    def run(ov, start):
        cfg = merged(BASE, ov)
        px = PX[PX.index >= pd.to_datetime(start)]
        e = SleeveEngine(px, config=cfg, yields=YL)
        R = e.rates_assets                       # 레그 공간: 실자산만
        pos = e.finalize_positions(e.compute_target_positions())

        dirr = e.dir_returns[R].reindex(pos.index).fillna(0.0)
        rets = yield_implied_returns(e, dirr)
        crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})
        turn = pos[R].diff().abs().fillna(0.0)

        def pnl(lag_map):
            held = pd.DataFrame({a: pos[a].shift(lag_map.get(a, 1))
                                 for a in R}).fillna(0.0)
            return (held * rets[R] - turn.mul(crate, axis=1)).sum(axis=1).dropna()

        s1 = pnl({a: 1 for a in R})
        s2 = pnl({a: 2 for a in R})
        stz = pnl({a: (2 if a in EARLY else 1) for a in R})
        st = perf_stats(s1)
        mid = len(s1) // 2
        return {
            'sr': st['sharpe'], 'ret': st['ret'], 'maxdd': st['maxdd'],
            'h1': perf_stats(s1.iloc[:mid])['sharpe'],
            'h2': perf_stats(s1.iloc[mid:])['sharpe'],
            't2': perf_stats(s2)['sharpe'], 'tz': perf_stats(stz)['sharpe'],
            'turn': float(turn.sum(axis=1).mean() * TRADING_DAYS),
            'pnl': s1, 'engine': e,
        }

    results = {}
    for start in ['2016-01-01', '2012-01-01']:
        print("\n" + "=" * 104)
        print(f"  커브+블록예산 A/B — 금리 북, {start[:4]}+  "
              f"(시간대정직 = 조기마감 T+2·미국 T+1, net of costs)")
        print("=" * 104)
        print(f"{'구성':<22}{'SR':>7}{'H1':>6}{'H2':>6}{'T+2':>6}"
              f"{'시간대정직':>10}{'AnnRet':>8}{'MaxDD':>8}{'회전':>7}")
        for name, ov in CASES:
            r = run(ov, start)
            results[(name, start)] = r
            print(f"{name:<22}{r['sr']:>7.2f}{r['h1']:>6.2f}{r['h2']:>6.2f}"
                  f"{r['t2']:>6.2f}{r['tz']:>10.2f}{r['ret']:>8.2%}"
                  f"{r['maxdd']:>8.1%}{r['turn']:>6.0f}x")

    # ── 커브 북 단독 진단 (합성 공간) + 비용 스윕 ─────────────────────────
    print("\n" + "=" * 104)
    print("  커브 북 단독 진단 (합성 스티프너 공간, 선물수익률 기반; "
          "비용 = 두 다리 (1+H)·|Δp|)")
    print("=" * 104)
    for start in ['2016-01-01', '2012-01-01']:
        cfg = merged(BASE, {'curve_trades': {'enabled': True}})
        px = PX[PX.index >= pd.to_datetime(start)]
        e = SleeveEngine(px, config=cfg, yields=YL)
        pos_syn = e.finalize_positions(e.compute_target_positions(),
                                       explode_curves=False)
        C = e.curve_assets
        if not C:
            print("  ⚠ 커브 자산 없음 — 진단 생략")
            continue
        rets = e.dir_returns[C].reindex(pos_syn.index).fillna(0.0)
        gross = (pos_syn[C].shift(1).fillna(0.0) * rets).sum(axis=1)
        # 두 다리 회전: 합성 |Δp| × (1 + H)
        legs_turn = pd.DataFrame(index=pos_syn.index)
        for name in C:
            _s, _l, H = e._curve_legs[name]
            legs_turn[name] = (pos_syn[name].diff().abs()
                               * (1.0 + H.reindex(pos_syn.index))).fillna(0.0)
        base_pnl = results[('A. 현행', start)]['pnl']
        rho = gross.corr(base_pnl.reindex(gross.index))
        line = f"  {start[:4]}+  corr(커브, 현행A 금리북) = {rho:5.2f}   "
        for bps in [0.5, 1.0, 2.0]:
            net = (gross - legs_turn.sum(axis=1) * bps / 10000.0).dropna()
            line += f"| SR@{bps}bp {perf_stats(net)['sharpe']:5.2f} "
        turn_ann = float(legs_turn.sum(axis=1).mean() * TRADING_DAYS)
        print(line + f"| 레그회전 {turn_ann:.0f}x/yr")

    print("\n  * 판정 기준(기존 프로토콜): 2016+·2012+ 모두, H1·H2·T+2·시간대정직")
    print("    전반 개선 + 커브 단독이 1bp 에서도 생존해야 채택. 같은 표본 재튜닝 금지.")


if __name__ == '__main__':
    main()
