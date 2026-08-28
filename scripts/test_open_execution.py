# -*- coding: utf-8 -*-
"""개장가 집행 측정 (2026-08-28) — 판정 게이트 없는 '측정/진단' 스크립트.

배경: 공식 백테스트는 종가 결정→종가 집행을 가정하지만 실제 운용은 아침에
시그널을 보고 장중(개장 부근)에 집행한다. 또한 2026-07-22 감사에서 아시아
종가가 미국 전일 움직임을 다음날 '장중에' 따라잡는다는 사실이 확인됐다
(KE1 일간수익률 vs 美전일 상관 0.33; 리버전 서브북은 그 캐치업을 종가에
소급 체결하는 아티팩트라 영구 OFF). 종가 프레임으로는 개장~종가 구간을
잴 수 없으므로 PX_OPEN 패널(data/cache/open_prices_*.parquet)로 두 가지를
측정한다:

  A. 현행 북의 개장가 집행 회계 — pos(t-1 종가 결정)를 t 개장에 체결:
       pnl_t = pos_{t-2}·r(종가→개장) + pos_{t-1}·r(개장→종가) − 비용
     공식(종가 체결) 회계와의 SR 격차 = 백테스트와 실집행의 괴리 추정치.
     ⚠ 원시 선물수익률 기준 (개장/종가 분해라 yield-implied 귀속 불가) —
     레벨이 공식 로그보다 ~0.1 낮은 것은 알려진 귀속 격차이며, 관심 대상은
     두 회계 방식의 '차이'다.

  B. 캐치업 진단 — 美 듀레이션 전일 수익률(z, ±2 클립)을 한국 개장에서
     잡아 종가에 청산(개장→종가 구간만 수확, 비용 왕복 2×0.5bp):
       美 마감 ~05-07시 KST → KTB 09:00 개장이라 물리적으로 집행 가능.
     대조군: 같은 시그널을 종가→익일종가로 들면 소멸해야 정상 (T+1 집행
     시 SR 1.01→−0.14 로 죽었던 리버전 감사 결과의 재현).

⚠ 이 스크립트는 채택/기각을 판정하지 않는다. B가 강하게 나오더라도 그건
'별도 사전등록 사이클 + 운영 설계(개장 주문 프로세스)'의 출발점일 뿐이며,
영구 OFF 인 리버전 서브북을 되살릴 근거가 아니다.

Usage: python scripts/test_open_execution.py
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

CACHE = Path(__file__).parent.parent / 'data' / 'cache'
DEV = ('2012-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)
TRADED = ['TU1 Comdty', 'TY1 Comdty', 'KE1 Comdty', 'KAA1 Comdty']
US = ['TU1 Comdty', 'TY1 Comdty']
KR = ['KE1 Comdty', 'KAA1 Comdty']


def window(s, a, b):
    s = s[s.index >= pd.to_datetime(a)]
    if b:
        s = s[s.index <= pd.to_datetime(b)]
    return s.dropna()


def line(tag, s, extra=''):
    st = perf_stats(s)
    print(f'  {tag:34s} SR {st["sharpe"]:5.2f} | AnnRet {st["ret"]:6.2%} | '
          f'MaxDD {st["maxdd"]:6.1%}{extra}')


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    MC = ld.load_signal_macro(start_date='2010-01-01', use_cache=True)
    cfg = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(cfg.get('costs_bps', {}) or {})}

    op_files = sorted(CACHE.glob('open_prices_*.parquet'))
    if not op_files:
        raise SystemExit('❌ open_prices 캐시 없음 (blp.bdh PX_OPEN 으로 재수집)')
    O_raw = pd.read_parquet(op_files[-1])

    e = SleeveEngine(PX, config=cfg, yields=YL, macro=MC)
    pos = e.finalize_positions(e.compute_target_positions())
    R = [a for a in TRADED if a in pos.columns]
    sign = e.sign[R]
    C = e.prices[R]
    O = O_raw.reindex(PX.index)[R].where(lambda x: x > 0)
    miss = O.isna() & C.notna()
    print(f'개장가 결측(자산·일 셀): {int(miss.sum().sum())} / {miss.size}'
          f'  (결측일은 공식 회계로 폴백)')

    r_cc = C.pct_change().mul(sign, axis=1)
    r_co = (O / C.shift(1) - 1.0).mul(sign, axis=1)
    r_oc = (C / O - 1.0).mul(sign, axis=1)
    # 결측 개장가 → 그 날은 공식(종가) 회계로 폴백
    r_co = r_co.where(~miss, r_cc)
    r_oc = r_oc.where(~miss, 0.0)
    chk = ((1 + r_co) * (1 + r_oc) - 1 - r_cc).abs().stack().mean()
    print(f'분해 정합성 |(1+co)(1+oc)-1-cc| 평균: {chk:.2e}')

    crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})
    turn = pos[R].diff().abs().fillna(0.0)

    # A. 회계 비교 (둘 다 원시 선물수익률)
    pnl_close = (pos[R].shift(1).fillna(0.0) * r_cc
                 - turn.mul(crate, axis=1)).sum(axis=1).dropna()
    pnl_open = (pos[R].shift(2).fillna(0.0) * r_co
                + pos[R].shift(1).fillna(0.0) * r_oc
                - turn.shift(1).fillna(0.0).mul(crate, axis=1)).sum(axis=1).dropna()

    print('\n' + '=' * 78)
    print('  A. 현행 북 — 종가 집행 vs 익일 개장 집행 (원시 선물수익률, net)')
    print('=' * 78)
    for tag, (a, b) in [('개발 2012-2021', DEV), ('홀드아웃 2022+', HOLD),
                        ('2016+', ('2016-01-01', None))]:
        s1, s2 = window(pnl_close, a, b), window(pnl_open, a, b)
        print(f'  [{tag}]')
        line('종가 집행 (공식 가정)', s1)
        line('개장 집행 (실운용 근사)', s2,
             f'  ΔSR {perf_stats(s2)["sharpe"] - perf_stats(s1)["sharpe"]:+.2f}')
    # 자산별 격차 (2016+, 비용 제외 gross 기여 차이)
    g_close = (pos[R].shift(1) * r_cc)
    g_open = (pos[R].shift(2) * r_co + pos[R].shift(1) * r_oc)
    diff = (g_open - g_close)
    diff16 = diff[diff.index >= '2016-01-01'].sum() * 100
    print('  자산별 누적 격차 (개장−종가, 2016+, gross %p):  '
          + ', '.join(f'{a.split()[0]} {diff16[a]:+.2f}' for a in R))

    # B. 캐치업 진단
    u = e.dir_returns[US].mean(axis=1)
    z = (u / u.rolling(252, min_periods=60).std()).clip(-2, 2)
    inv_vol = (e.target_asset_vol / e._realized_vol()[KR]).clip(upper=15.0)
    # 인덱스 정렬: 패널 행 t 의 KR 개장(09:00 KST)은 美 행 t-1 마감(같은 날
    # 05-07시 KST) 이후다 → 행 t 포지션은 z_{t-1} 사용.
    p = inv_vol.mul(z.shift(1), axis=0)
    pnl_b = (p * r_oc[KR] - 2.0 * p.abs().mul(crate[KR], axis=1)).sum(axis=1).dropna()
    gross_b = (p * r_oc[KR]).sum(axis=1).dropna()
    # 대조군: 같은 시그널을 종가→익일종가 보유 (아티팩트 소멸 확인용)
    pnl_ctrl = (p.shift(1) * r_cc[KR]
                - 2.0 * p.shift(1).abs().mul(crate[KR], axis=1)).sum(axis=1).dropna()

    print('\n' + '=' * 78)
    print('  B. 캐치업 진단 — z(美 전일) → 한국 개장 진입/종가 청산 (왕복 1bp)')
    print('=' * 78)
    for a in KR:
        both = pd.concat([u.shift(1), r_oc[a]], axis=1, keys=['u', 'r']).dropna()
        both = both[both.index >= pd.to_datetime(DEV[0])]
        print(f'  IC corr(美 전일, {a.split()[0]} 개장→종가): {both["u"].corr(both["r"]):+.3f}')
    for tag, (a, b) in [('개발 2012-2021', DEV), ('홀드아웃 2022+', HOLD)]:
        print(f'  [{tag}]')
        line('개장→종가 gross', window(gross_b, a, b))
        line('개장→종가 net', window(pnl_b, a, b))
        line('대조군: 종가→익일종가 net', window(pnl_ctrl, a, b),
             '  (≈0 이어야 리버전 감사와 정합)')

    print('\n  * 측정 전용 — 채택 판정 없음. B 를 살리려면 별도 사전등록 사이클'
          ' + 개장 주문 운영 설계가 선행되어야 한다.')


if __name__ == '__main__':
    main()
