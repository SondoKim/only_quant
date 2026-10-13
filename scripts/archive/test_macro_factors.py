# -*- coding: utf-8 -*-
"""매크로 데이터 슬리브(inflation/path) 홀드아웃 A/B (2026-08-14).

배경 (memory rates-review-2026-08): 북의 실질 독립 팩터는 '방향성 금리 모멘텀'
하나 (trend↔policy 상관 0.84). 개선 병목은 팩터 개수가 아니라 새 정보원이므로
가격·금리 밖의 데이터로 두 슬리브를 검증한다:
  inflation — 인플레 기대 모멘텀 (US/UK/JP 10Y BEI 63d 변화, KR/AU CPI YoY
              126d 변화 + 발표랙 25거래일). 상승 → 숏 듀레이션.
  path      — 정책 경로 리프라이싱 (gap = 1Y OIS − 정책금리, Δ63d).
              완화를 '새로' 프라이싱 → 롱 듀레이션.
둘 다 방향성(xs 0) → 실질 영향은 매매 4종(韓美)뿐. 가중은 사전등록 1.0(균등
합류), 파라미터 재튜닝 금지.

⚠ 홀드아웃 프로토콜 — 2016+ 전표본은 수십 번의 A/B 로 이미 오염됐다:
  신호는 전체 히스토리로 계산(워밍업 왜곡 방지)하되, 판정은
    개발표본  2012-01-01 ~ 2021-12-31   에서 내리고
    홀드아웃  2022-01-01 ~              은 확인용으로만 본다.
  채택 게이트 (사전등록):
    G1 기존 슬리브와의 시그널 상관 |ρ| < 0.5 (개발표본, 매매 자산 평균)
       — 0.5 이상이면 trend/policy 재판 (기각).
    G2 개발표본 전 열(SR/H1/H2/T+2/시간대정직) 개선.
    G3 홀드아웃 SR 이 베이스라인 −0.03 이상 (악화 없음 확인).

Usage: python scripts/test_macro_factors.py
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

DEV = ('2012-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)

CASES = [
    ('A. 현행',            {}),
    ('B. +inflation',      {'inflation': 1.0}),
    ('C. +path',           {'path': 1.0}),
    ('D. +both',           {'inflation': 1.0, 'path': 1.0}),
]


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
    MC = ld.load_signal_macro(start_date='2010-01-01', use_cache=True)
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}

    def build(w_ov):
        cfg = {**BASE}
        cfg['sleeve_weights'] = {**(BASE.get('sleeve_weights') or {}), **w_ov}
        return SleeveEngine(PX, config=cfg, yields=YL, macro=MC)

    # ── G1: 시그널 상관 게이트 (개발표본, 매매 자산) ──────────────────
    e0 = build({'inflation': 1.0, 'path': 1.0})
    traded = [a for a in e0.rates_assets if a not in e0.signal_only_assets]
    exist = {'trend': e0.trend_signal(e0.rates_assets),
             'value': e0.value_signal(e0.rates_assets, 'rates'),
             'carry': e0.carry_signal(e0.rates_assets, 'rates'),
             'policy': e0.policy_signal(e0.rates_assets)}
    new = {'inflation': e0.inflation_signal(e0.rates_assets),
           'path': e0.path_signal(e0.rates_assets)}
    print("=" * 78)
    print("  G1  시그널 상관 (개발표본 2012-2021, 매매 자산 평균) — |ρ|<0.5 필요")
    print("=" * 78)
    gate1 = {}
    for nn, ns in new.items():
        row = []
        for en, es in exist.items():
            cs = []
            for a in traded:
                x = window(ns[a], *DEV)
                y = window(es[a], *DEV)
                c = x.corr(y)
                if not np.isnan(c):
                    cs.append(c)
            m = float(np.mean(cs)) if cs else np.nan
            row.append((en, m))
        worst = max(abs(m) for _, m in row if not np.isnan(m))
        gate1[nn] = worst < 0.5
        cell = '  '.join(f"{en} {m:+.2f}" for en, m in row)
        print(f"  {nn:10s} {cell}   → max|ρ| {worst:.2f} "
              f"{'PASS' if gate1[nn] else 'FAIL'}")
    # 신규 슬리브끼리
    cs = [window(new['inflation'][a], *DEV).corr(window(new['path'][a], *DEV))
          for a in traded]
    print(f"  inflation vs path: {np.nanmean(cs):+.2f}")

    # ── 성과 A/B ─────────────────────────────────────────────────────
    def run(w_ov):
        e = build(w_ov)
        R = e.rates_assets
        pos = e.finalize_positions(e.compute_target_positions())
        dirr = e.dir_returns[list(pos.columns)].reindex(pos.index).fillna(0.0)
        rets = yield_implied_returns(e, dirr)
        crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})
        turn = pos[R].diff().abs().fillna(0.0)

        def pnl(lag_map):
            held = pd.DataFrame({a: pos[a].shift(lag_map.get(a, 1))
                                 for a in R}).fillna(0.0)
            return (held * rets[R]
                    - turn.mul(crate, axis=1)).sum(axis=1).dropna()

        return {'t1': pnl({a: 1 for a in R}),
                't2': pnl({a: 2 for a in R}),
                'tz': pnl({a: (2 if a in EARLY else 1) for a in R}),
                'turn': turn.sum(axis=1)}

    def report(tag, a, b):
        print("\n" + "=" * 96)
        print(f"  {tag}")
        print("=" * 96)
        print(f"{'구성':<16}{'SR':>7}{'H1':>6}{'H2':>6}{'T+2':>6}"
              f"{'시간대정직':>10}{'AnnRet':>8}{'MaxDD':>8}{'회전':>7}")
        out = {}
        for name, w_ov in CASES:
            r = res[name]
            s1 = window(r['t1'], a, b)
            st = perf_stats(s1)
            mid = len(s1) // 2
            out[name] = st['sharpe']
            print(f"{name:<16}{st['sharpe']:>7.2f}"
                  f"{perf_stats(s1.iloc[:mid])['sharpe']:>6.2f}"
                  f"{perf_stats(s1.iloc[mid:])['sharpe']:>6.2f}"
                  f"{perf_stats(window(r['t2'], a, b))['sharpe']:>6.2f}"
                  f"{perf_stats(window(r['tz'], a, b))['sharpe']:>10.2f}"
                  f"{st['ret']:>8.2%}{st['maxdd']:>8.1%}"
                  f"{window(r['turn'], a, b).mean() * TRADING_DAYS:>6.0f}x")
        return out

    res = {name: run(w_ov) for name, w_ov in CASES}
    dev = report("개발표본 2012-2021 — 판정은 여기서만", *DEV)
    hold = report("홀드아웃 2022+ — 확인 전용 (여기 보고 고르면 홀드아웃이 아니다)", *HOLD)

    print("\n  게이트 요약: G1 상관 — inflation "
          f"{'PASS' if gate1.get('inflation') else 'FAIL'}, "
          f"path {'PASS' if gate1.get('path') else 'FAIL'}")
    print("  G2 개발표본 전 열 개선 + G3 홀드아웃 ΔSR ≥ -0.03 은 위 표로 판정.")
    print("  * 신호는 전 히스토리로 계산, 평가 구간만 분리 (워밍업 왜곡 방지).")


if __name__ == '__main__':
    main()
