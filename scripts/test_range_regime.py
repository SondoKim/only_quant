# -*- coding: utf-8 -*-
"""박스권 레짐 게이트 + 챱(횡보) 북 A/B (2026-09-03, 사용자 제안).

제안: 챱 구간에서 작동하는 매매 전략을 갖추고, '박스권 유지 기간'을 지표로
박스가 계속 이동하면 추세장 / 정체되면 횡보장으로 나눠 챱 전략 비중을
켜고 끈다.

구현 (엔진 range_regime, 기본 OFF):
  감지기  stale = W일 신고/신저 없이 지난 일수 ≥ K → 횡보(g=1). scope 자산별/북.
  챱 북   channel = 박스 중앙 대비 위치 페이드 (상단 매도/하단 매수), value_ts =
          방향성 밸류.
  결합    blend = (1−w·g)·main + w·g·chop / add = main + w·g·chop. w=0.5 사전등록.

측정 (두 질문을 분리):
  Q1 감지기에 예측력이 있는가 — g(t−1)=1 인 날과 0 인 날의 현행 북 SR 비교.
     횡보 표시 뒤 추세 북이 실제로 못 벌어야 게이트가 성립한다.
  Q2 챱 북이 감지된 횡보에서 버는가 — 챱 북 단독(always_on, w=1) PnL 을 같은
     버킷으로.
  A/B 는 test_switching.py 와 동일 프로토콜 (개발 2012-21 판정 / 홀드아웃
  2022+ 확인, yield-implied, net, T+2·시간대정직). 격자는 사전등록 W/K ∈
  {63/21, 126/42}, w=0.5, 사후 세분화 금지.

전례: Hurst 레짐 게이트 = 죽은 복잡성(2026-08-14 제거). 리버전 서브북 = 비동시
종가 아티팩트(2026-07-22) — 평균회귀 계열은 T+2 열이 결정적이다.

Usage: python scripts/test_range_regime.py
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
from scripts.test_curve_blocks import yield_implied_returns, merged  # noqa: E402

DEV = ('2012-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)
EARLY = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
         'YM1 Comdty', 'XM1 Comdty']
TOL = 0.03


def rr(**kw):
    base = {'enabled': True, 'window': 63, 'stale_days': 21, 'chop_weight': 0.5,
            'chop_book': 'channel', 'scope': 'asset', 'mode': 'blend', 'smooth': 5}
    return {'range_regime': {**base, **kw}}


CASES = [
    ('A. 현행',                          {}),
    ('R0. channel 상시 blend .5',        rr(always_on=True)),
    ('R1. channel asset 63/21 blend',    rr()),
    ('R2. channel asset 126/42 blend',   rr(window=126, stale_days=42)),
    ('R3. channel book 63/21 blend',     rr(scope='book')),
    ('R4. channel asset 63/21 add',      rr(mode='add')),
    ('R5. value_ts asset 63/21 blend',   rr(chop_book='value_ts')),
    ('R6. value_ts book 63/21 blend',    rr(chop_book='value_ts', scope='book')),
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
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}
    FX_PIN = dict(BASE.get('xs_neutralize') or
                  {'trend': 0.0, 'value': 1.0, 'carry': 1.0})

    def build(ov):
        cfg = merged(BASE, ov)
        cfg['xs_neutralize_fx'] = FX_PIN
        return SleeveEngine(PX, config=cfg, yields=YL)

    def pnl_frames(e):
        """자산별 일별 순손익 (T+1 / T+2 / 시간대정직)."""
        R = e.rates_assets
        pos = e.finalize_positions(e.compute_target_positions())
        dirr = e.dir_returns[R + e.fx_assets].reindex(pos.index).fillna(0.0)
        rets = yield_implied_returns(e, dirr)
        crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})
        turn = pos[R].diff().abs().fillna(0.0)

        def per_asset(lag_map):
            held = pd.DataFrame({a: pos[a].shift(lag_map.get(a, 1)) for a in R}).fillna(0.0)
            return held * rets[R] - turn.mul(crate, axis=1)

        return {'t1': per_asset({a: 1 for a in R}),
                't2': per_asset({a: 2 for a in R}),
                'tz': per_asset({a: (2 if a in EARLY else 1) for a in R}),
                'turn': turn.sum(axis=1), 'pos': pos}

    # ── Q1/Q2: 감지기 예측력 + 챱 북 조건부 수익 ───────────────────────
    print('=' * 96)
    print('  Q1 감지기 예측력 / Q2 챱 북 조건부 수익 — g(t−1) 버킷별 연환산 SR '
          '(매매 4종 합, T+1 net)')
    print('=' * 96)
    eA = build({})
    T = [a for a in eA.rates_assets if a not in eA.signal_only_assets]
    fA = pnl_frames(eA)
    chop_books = {'channel': build(rr(always_on=True, chop_weight=1.0)),
                  'value_ts': build(rr(always_on=True, chop_weight=1.0,
                                       chop_book='value_ts'))}
    fC = {k: pnl_frames(v) for k, v in chop_books.items()}

    def bucket_sr(frame, g, a_, b_):
        s = window(frame[T].sum(axis=1), a_, b_)
        gl = g.shift(1).reindex(s.index)
        out = {}
        for lbl, m in [('횡보', gl >= 0.5), ('추세', gl < 0.5)]:
            x = s[m.fillna(False)]
            out[lbl] = (perf_stats(x)['sharpe'] if len(x) > 60 else np.nan, len(x))
        return out

    # 버킷은 매매 4종 g 의 북 평균(≥0.5 = 횡보) — 자산별 레짐(scope asset)과
    # 북 레짐(scope book)의 차이는 아래 A/B 행(R1 vs R3)에서 본다.
    print(f"{'감지기':<16}{'구간':<12}{'횡보일%':>8}{'현행북 횡보':>12}{'현행북 추세':>12}"
          f"{'channel 횡보':>13}{'channel 추세':>13}{'value_ts 횡보':>14}{'value_ts 추세':>14}")
    for W, K in [(63, 21), (126, 42)]:
        ed = build(rr(window=W, stale_days=K, smooth=1))
        g = ed.range_regime_indicator(ed.rates_assets)[T].mean(axis=1)
        for tag, (a_, b_) in [('dev 12-21', DEV), ('hold 22+', HOLD)]:
            share = float((window(g, a_, b_) >= 0.5).mean())
            bA = bucket_sr(fA['t1'], g, a_, b_)
            bC = bucket_sr(fC['channel']['t1'], g, a_, b_)
            bV = bucket_sr(fC['value_ts']['t1'], g, a_, b_)
            print(f"{f'W{W}/K{K}':<16}{tag:<12}{share:>8.0%}"
                  f"{bA['횡보'][0]:>12.2f}{bA['추세'][0]:>12.2f}"
                  f"{bC['횡보'][0]:>13.2f}{bC['추세'][0]:>13.2f}"
                  f"{bV['횡보'][0]:>14.2f}{bV['추세'][0]:>14.2f}")
    print('  * 게이트 성립 조건: 현행북 SR 이 횡보 버킷에서 낮고, 챱 북 SR 이 횡보 버킷에서'
          ' 높아야 한다 (둘 다 필요).')
    print('  * 2026-09-03 결과: W63/K21 은 현행북이 횡보 버킷에서 오히려 더 벌고(1.83 vs'
          ' 0.00), 챱 북은 모든 버킷에서 음수 — 두 전제 모두 불성립.')
    for k, f in fC.items():
        for tag, (a_, b_) in [('dev', DEV), ('hold', HOLD)]:
            s1 = window(f['t1'][T].sum(axis=1), a_, b_)
            s2 = window(f['t2'][T].sum(axis=1), a_, b_)
            print(f"  챱 북 단독 {k:<9}{tag:<5} SR T+1 {perf_stats(s1)['sharpe']:5.2f} "
                  f"| T+2 {perf_stats(s2)['sharpe']:5.2f} | 회전 "
                  f"{window(f['turn'], a_, b_).mean() * TRADING_DAYS:4.0f}x")

    # ── A/B ─────────────────────────────────────────────────────────────
    def run(ov):
        e = build(ov)
        f = pnl_frames(e)
        R = e.rates_assets
        s1 = f['t1'][R].sum(axis=1).dropna()
        s2 = f['t2'][R].sum(axis=1).dropna()
        stz = f['tz'][R].sum(axis=1).dropna()
        d1 = window(s1, *DEV)
        mid = len(d1) // 2
        st = {
            'dev': perf_stats(d1)['sharpe'],
            'h1': perf_stats(d1.iloc[:mid])['sharpe'],
            'h2': perf_stats(d1.iloc[mid:])['sharpe'],
            't2': perf_stats(window(s2, *DEV))['sharpe'],
            'tz': perf_stats(window(stz, *DEV))['sharpe'],
            'dd': perf_stats(d1)['maxdd'],
            's16': perf_stats(window(s1, '2016-01-01', None))['sharpe'],
            'hold': perf_stats(window(s1, *HOLD))['sharpe'],
            'hdd': perf_stats(window(s1, *HOLD))['maxdd'],
            'turn': float(window(f['turn'], *DEV).mean() * TRADING_DAYS),
        }
        if e.range_enabled and e._range_state is not None and not e.range_always_on:
            g = e._range_state['g'][T].mean(axis=1)
            st['gshare'] = float((window(g, *DEV) >= 0.5).mean())
        else:
            st['gshare'] = np.nan
        return st

    GATE = ['dev', 'h1', 'h2', 't2', 'tz']
    hdr = (f"{'구성':<34}{'devSR':>7}{'H1':>6}{'H2':>6}{'T+2':>6}{'정직':>6}"
           f"{'devDD':>7}{'2016+':>7}{'hold':>6}{'hDD':>7}{'회전':>6}{'횡보%':>6}")
    print()
    print('=' * len(hdr))
    print('  A/B — 개발 2012-2021 판정 / 홀드아웃 2022+ 확인 (탈락 구성 hold 비공개)')
    print('=' * len(hdr))
    print(hdr)
    base = None
    passers = []
    for name, ov in CASES:
        r = run(ov)
        if base is None:
            base, ok = r, True
        else:
            ok = all(r[c] >= base[c] - TOL for c in GATE)
        hold = f"{r['hold']:>6.2f}{r['hdd']:>7.1%}" if ok else f"{'—':>6}{'—':>7}"
        gs = f"{r['gshare']:>6.0%}" if not np.isnan(r['gshare']) else f"{'—':>6}"
        print(f"{('✓ ' if ok and base is not r else '  ') + name:<34}"
              f"{r['dev']:>7.2f}{r['h1']:>6.2f}{r['h2']:>6.2f}{r['t2']:>6.2f}"
              f"{r['tz']:>6.2f}{r['dd']:>7.1%}{r['s16']:>7.2f}{hold}"
              f"{r['turn']:>5.0f}x{gs}")
        if ok and base is not r:
            passers.append((name, r))
    print()
    if not passers:
        print('  → 개발 게이트 통과 구성 없음: 현행 유지.')
    else:
        for name, r in passers:
            v = ('홀드아웃 통과' if r['hold'] >= base['hold'] - TOL else '홀드아웃 탈락')
            print(f"  → {name}: dev {r['dev']:.2f} (베이스 {base['dev']:.2f}), "
                  f"hold {r['hold']:.2f} (베이스 {base['hold']:.2f}) — {v}")
        print('  ⚠ 홀드아웃 재사용 상태 — 아슬아슬한 통과는 채택하지 않는다.')


if __name__ == '__main__':
    main()
