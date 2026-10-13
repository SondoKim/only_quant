# -*- coding: utf-8 -*-
"""슬리브 구성 · 캐리 산식 재검증 (2026-10-13, 사전등록).

배경 (2026-10-13 감사):
  · 10-02 에 채택한 hedged_xs 캐리의 우위(SR 0.63→0.72)는 ASX(YM/XM, 100−금리 호가)의 위험
    분모 오류 덕이었다 — 가격 수익률 변동성이 듀레이션 ~1 처럼 나와 XM 캐리/위험이 ~8배 과대,
    횡단면 평균을 지배해 韓美 캐리에 우연한 방향성 틸트가 섞였다. 교정하면 0.58.
  · 2026-08 의 '4슬리브 균등 확정'은 롤 아티팩트·yield_implied 손익 시절 결론이라 무효.
  · 슬리브 단독(무스톱, 2016+): trend 0.39 · value −0.18 · carry −0.17 · policy 0.58.

후보 (사전등록 — 추가·수정 금지). 캐리는 모두 교정판(C* = ASX 를 par 듀레이션×금리볼로):
  A  trend + value + carry* + policy
  B  trend + value + policy
  C  trend + carry* + policy
  D  trend + policy
  (참고, 판정 제외) S0 = 현행 왜곡 캐리 4슬리브 — 알려진 결함이라 후보 자격 없음.

평가: 롤 보정·futures 손익·비용 차감, 엔진은 2010-10 패널 전체 워밍업.
  개발 2012-01-01~2021-12-31 · 홀드아웃 2022-01-01~ · 전체 2012~ .

판정 규칙 (사전등록):
  1) 자격: 전체 T+2 SR ≥ T+1 SR − 0.10, 시간대정직(韓 T+2) SR ≥ T+1 SR − 0.10, 홀드아웃 SR > 0.
  2) '잡음 범위 내' = 개발 SR ≥ (자격자 최고 개발 SR − 0.10) AND 홀드아웃 SR ≥ (자격자 최고 홀드 − 0.10).
     (10년 SR 표준오차 ≈ ±0.3 이므로 0.10 은 보수적인 동등성 폭)
  3) 잡음 범위 내 후보 중 슬리브 수가 가장 적은 것, 동수면 전체 SR 이 높은 것을 채택.
  잡음 범위 내 후보가 없으면(=자격자 없음) 현행 유지 + 보고.

Usage: python scripts/research/test_composition_2026_10.py
"""
import sys
import logging
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
logging.disable(logging.WARNING)

from src.data.loader import DataLoader                             # noqa: E402
from src.sleeves.sleeve_engine import SleeveEngine                 # noqa: E402
from src.sleeves.accounting import book_pnl, sharpe, window, max_drawdown  # noqa: E402
from scripts.run_sleeve_backtest import load_sleeve_config         # noqa: E402

DEV = ('2012-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)
FULL = ('2012-01-01', None)
KR = {'KE1 Comdty', 'KAA1 Comdty'}
ASX_TENOR = {'YM1 Comdty': 3, 'XM1 Comdty': 10}

CANDS = {
    'A T+V+C*+P': ['trend', 'value', 'carry', 'policy'],
    'B T+V+P':    ['trend', 'value', 'policy'],
    'C T+C*+P':   ['trend', 'carry', 'policy'],
    'D T+P':      ['trend', 'policy'],
}


def main():
    ld = DataLoader()
    base = load_sleeve_config()
    px, yl, mc = ld.engine_inputs(base)
    rows = {}

    def evaluate(name, sleeves, asx_fix):
        cfg = {**base, 'sleeve_weights': {k: (1.0 if k in sleeves else 0.0)
                                          for k in ('trend', 'value', 'carry', 'policy')}}
        e = SleeveEngine(px, config=cfg, yields=yl, macro=mc)
        e.yield_quoted_tenor = dict(ASX_TENOR) if asx_fix else {}
        p = e.finalize_positions(e.compute_target_positions())
        R = [a for a in e.rates_assets if a in p.columns]
        q1 = book_pnl(e, p, R)
        q2 = book_pnl(e, p, R, lag_map={a: 2 for a in R})
        qz = book_pnl(e, p, R, lag_map={a: (2 if a in KR else 1) for a in R})
        rows[name] = {'슬리브수': len(sleeves),
                      'SR 개발': sharpe(window(q1, *DEV)), 'SR 홀드': sharpe(window(q1, *HOLD)),
                      'SR 전체': sharpe(window(q1, *FULL)), 'T+2': sharpe(window(q2, *FULL)),
                      '시간대정직': sharpe(window(qz, *FULL)),
                      'MaxDD%': max_drawdown(window(q1, *FULL)) * 100,
                      '연수익%': window(q1, *FULL).mean() * 252 * 100,
                      '오늘 TU': p['TU1 Comdty'].iloc[-1], '오늘 TY': p['TY1 Comdty'].iloc[-1],
                      '오늘 KE': p['KE1 Comdty'].iloc[-1], '오늘 KAA': p['KAA1 Comdty'].iloc[-1]}

    evaluate('S0 현행(왜곡캐리,참고)', ['trend', 'value', 'carry', 'policy'], asx_fix=False)
    for name, sl in CANDS.items():
        evaluate(name, sl, asx_fix=True)
    df = pd.DataFrame(rows).T
    pd.set_option('display.width', 220)
    print(df.to_string(float_format=lambda x: f"{x:6.2f}"))

    c = df.loc[list(CANDS)]
    elig = c[(c['T+2'] >= c['SR 전체'] - 0.10) & (c['시간대정직'] >= c['SR 전체'] - 0.10) & (c['SR 홀드'] > 0)]
    print(f"\n자격 통과: {list(elig.index)}")
    if elig.empty:
        print("판정: 자격자 없음 → 현행 유지")
        return
    near = elig[(elig['SR 개발'] >= elig['SR 개발'].max() - 0.10) & (elig['SR 홀드'] >= elig['SR 홀드'].max() - 0.10)]
    print(f"잡음 범위 내 (최고 개발 {elig['SR 개발'].max():.2f}, 최고 홀드 {elig['SR 홀드'].max():.2f}, 폭 0.10): {list(near.index)}")
    if near.empty:
        print("판정: 잡음 범위 내 후보 없음 → 현행 유지")
        return
    pick = near.sort_values(['슬리브수', 'SR 전체'], ascending=[True, False]).index[0]
    print(f"판정(사전등록 규칙): {pick} — 슬리브 {CANDS[pick]}")


if __name__ == '__main__':
    main()
