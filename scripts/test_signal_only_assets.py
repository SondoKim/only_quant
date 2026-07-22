# -*- coding: utf-8 -*-
"""집행 유니버스 A/B — 시그널은 전 시장, 매매는 집행 가능한 시장만.

동기 (사용자, 한국 소재): 한국·미국 국채선물은 집행 시간이 깨끗하지만
(KTB 09:00-15:45 KST 주간장 직접 집행 / 미국은 CME Globex 가 한국 주간에도
열려 있음), 유럽·영국·호주·일본은 매매 시간이 애매하다.

두 가지 방식이 다르다는 점이 핵심:
  · signal_only_assets — 시그널 유니버스에 남기고 포지션만 0. 횡단면 z·demean
    기준선은 전 시장 유지. value/carry(xs_neutralize=1.0)는 반대편 다리가
    사라져 '반쪽 헤지'가 되지만, trend/policy(xs_neutralize=0)는 순수 시계열
    이라 영향이 없다 — 그리고 알파는 대부분 후자에서 나온다([[directionality-ab]]).
  · exclude_assets — 시그널에서도 삭제. 횡단면이 4종으로 좁아진다.

⚠ 2026-07-21 에 이 스크립트의 이전 버전이 "signal_only 는 손해"라고 결론냈으나,
그 테스트는 리버전 아티팩트가 켜지고 book_stop 이 래치된 상태였다
([[stale-price-audit]]). 그 결론은 무효이며 아래가 정직한 재측정이다.

시차 감사 병기 필수: T+2 와 '시간대 정직'(조기마감 시장 T+2, 미국 T+1) SR 을
같이 본다. 한국을 한국 종가에 집행하면서 같은 날짜 미국 종가를 쓰는 것은
구조적 미래참조이므로, 시간대 정직 열이 실제로 집행 가능한 수치에 가깝다.

Usage: python scripts/test_signal_only_assets.py
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

KR_US = ['TU1 Comdty', 'TY1 Comdty', 'KE1 Comdty', 'KAA1 Comdty']
AWKWARD = ['G 1 Comdty', 'JB1 Comdty', 'YM1 Comdty', 'XM1 Comdty']  # 英·日·豪
EUROZONE = ['DU1 Comdty', 'RX1 Comdty', 'OAT1 Comdty', 'IK1 Comdty']  # 獨2·佛·伊
EARLY = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
         'YM1 Comdty', 'XM1 Comdty'] + EUROZONE   # 미국보다 먼저 마감

# 유로존은 2026-07-14 A/B 에서 '매매하면' 손해라 exclude 됐다. 하지만 매매와
# 시그널은 별개 질문이다 — 횡단면 demean 기준선(글로벌 듀레이션 평균)이 넓어지는
# 것 자체는 value/carry 의 시그널 품질을 높일 수 있다. E·F 가 그 검증.
CASES = [
    ('A. 현행 — 시그널8 / 매매8',          {}),
    ('B. 시그널8 / 매매 韓美4',            {'signal_only_assets': AWKWARD}),
    ('C. 시그널4 / 매매 韓美4',            {'exclude_extra': AWKWARD}),
    ('D. 시그널8 / 매매 韓美日5',          {'signal_only_assets':
                                          ['G 1 Comdty', 'YM1 Comdty', 'XM1 Comdty']}),
    ('E. 시그널12 / 매매 韓美4',           {'exclude_assets': [],
                                          'signal_only_assets': AWKWARD + EUROZONE}),
    ('F. 시그널12 / 매매8',                {'exclude_assets': [],
                                          'signal_only_assets': EUROZONE}),
    ('G. 시그널12 / 매매12',               {'exclude_assets': []}),
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
        extra = ov.pop('exclude_extra', [])
        cfg.update(ov)
        if extra:
            cfg['exclude_assets'] = list(cfg.get('exclude_assets', [])) + extra
        px = PX[PX.index >= pd.to_datetime(start)]
        e = SleeveEngine(px, config=cfg, yields=YL)
        R = e.rates_assets                       # 시그널 유니버스
        pos = e.finalize_positions(e.compute_target_positions())
        traded = [a for a in R if pos[a].abs().max() > 1e-9]

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
        s2 = pnl({a: 2 for a in R})
        stz = pnl({a: (2 if a in EARLY else 1) for a in R})
        st = perf_stats(s1)
        mid = len(s1) // 2
        return {
            'n': len(traded), 'sr': st['sharpe'], 'ret': st['ret'],
            'maxdd': st['maxdd'],
            'h1': perf_stats(s1.iloc[:mid])['sharpe'],
            'h2': perf_stats(s1.iloc[mid:])['sharpe'],
            't2': perf_stats(s2)['sharpe'], 'tz': perf_stats(stz)['sharpe'],
            'turn': float(turn.sum(axis=1).mean() * TRADING_DAYS),
        }

    for start in ['2016-01-01', '2012-01-01']:
        print("\n" + "=" * 100)
        print(f"  집행 유니버스 A/B — {start[:4]}+   (시간대정직 = 조기마감 T+2·미국 T+1)")
        print("=" * 100)
        print(f"{'구성':<26}{'매매':>4}{'SR':>7}{'H1':>6}{'H2':>6}"
              f"{'T+2':>6}{'시간대정직':>10}{'AnnRet':>8}{'MaxDD':>8}{'회전':>7}")
        for name, ov in CASES:
            r = run(ov, start)
            print(f"{name:<26}{r['n']:>4}{r['sr']:>7.2f}{r['h1']:>6.2f}{r['h2']:>6.2f}"
                  f"{r['t2']:>6.2f}{r['tz']:>10.2f}{r['ret']:>8.2%}"
                  f"{r['maxdd']:>8.1%}{r['turn']:>6.0f}x")

    print("\n  * B vs C 의 차이 = '호주·일본·영국을 횡단면 기준선에 남겨두는 값어치'.")
    print("  * 한국을 한국 종가에 집행하며 당일 미국 종가를 쓰는 것은 구조적 미래참조 —")
    print("    실제 집행 가능한 수치는 '시간대정직' 열에 가깝다.")


if __name__ == '__main__':
    main()
