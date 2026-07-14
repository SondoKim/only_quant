# -*- coding: utf-8 -*-
"""금리 유니버스 축소 A/B — 어떤 자산을 빼면 운용이 단순해지고 성과가 유지/개선되나.

각 조합을 동일 조건(2016+, 프로덕션 config)으로 재실행 — 자산 제외 시
xs-demean·볼타겟팅이 전부 재계산되므로 단순 기여도 뺄셈과 다름.
과최적화 방지: 전/후반 분할 샤프를 함께 보고, 양쪽 모두 유지되는 조합만 신뢰.

Usage: python scripts/test_universe_reduction.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_sleeve_backtest import run, perf_stats  # noqa: E402

START = '2016-01-01'

# (이름, 제외 자산) — 운용 단순화 논리 기준의 후보들
CONFIGS = [
    ('A. 현행 12개', []),
    ('B. 독일 제외 (10개)', ['DU1 Comdty', 'RX1 Comdty']),
    ('C. 유로존 제외 (8개)', ['DU1 Comdty', 'RX1 Comdty', 'OAT1 Comdty', 'IK1 Comdty']),
    ('D. 10Y만 (8개, 단기테너 제외)', ['TU1 Comdty', 'DU1 Comdty', 'YM1 Comdty', 'KE1 Comdty']),
    ('E. 기여 상위6 (美10·佛10·日·豪10·韓3·韓10)',
     ['TU1 Comdty', 'DU1 Comdty', 'RX1 Comdty', 'IK1 Comdty', 'G 1 Comdty', 'YM1 Comdty']),
    ('F. 시장당 10Y 1개 (6개: 美·獨·英·日·豪·韓)',
     ['TU1 Comdty', 'DU1 Comdty', 'OAT1 Comdty', 'IK1 Comdty', 'YM1 Comdty', 'KE1 Comdty']),
]


def half_stats(s: pd.Series) -> tuple:
    s = s.dropna()
    mid = len(s) // 2
    return perf_stats(s.iloc[:mid])['sharpe'], perf_stats(s.iloc[mid:])['sharpe']


def main():
    results = []
    for name, excl in CONFIGS:
        print(f"\n{'='*70}\n▶ {name}\n{'='*70}")
        df = run(start_date=START, plot=False, save_outputs=False,
                 exclude_assets=excl or None)
        rates = df['rates'].dropna()
        st = perf_stats(rates)
        h1, h2 = half_stats(rates)
        results.append({
            '조합': name, '자산수': 12 - len(excl),
            'SR': st['sharpe'], 'H1': h1, 'H2': h2,
            'Vol': st['vol'], 'AnnRet': st['ret'], 'MaxDD': st['maxdd'],
        })

    print(f"\n\n{'='*100}")
    print("  금리 북 유니버스 축소 비교  (2016+ / 전·후반 분할 샤프)")
    print(f"{'='*100}")
    print(f"{'조합':<42}{'자산':>4} {'SR':>6} {'H1':>6} {'H2':>6} "
          f"{'Vol':>7} {'AnnRet':>8} {'MaxDD':>8}")
    for r in results:
        print(f"{r['조합']:<42}{r['자산수']:>4} {r['SR']:>6.2f} {r['H1']:>6.2f} "
              f"{r['H2']:>6.2f} {r['Vol']:>7.1%} {r['AnnRet']:>8.1%} {r['MaxDD']:>8.1%}")


if __name__ == '__main__':
    main()
