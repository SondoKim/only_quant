# -*- coding: utf-8 -*-
"""북스톱 구간 대체 북 연구 (2026-10-02, 사용자 요청: "스톱 구간에도 돈 버는 거래").

진단 (2016+, 롤보정·futures 손익·hedged_xs 기준):
  · 스톱(×0.5 또는 ×0) 상태 = 전체 일수의 54%. 스톱 없는 북의 그 구간 손익 −2.47%
    (SR −0.12) → 스톱은 평균적으로 손실을 막았다 (전체 SR 0.72 vs 무스톱 0.31).
  · 다만 연도별로 극단적: 2024 −10.4% 회피 vs 2018 +7.3%·2021 +4.7% 반등 놓침.
  → '메인 북을 되살리기'가 아니라 '스톱으로 비는 리스크 예산에 성격이 다른 북을 넣기'.

결합 규칙: 최종 = 스톱 적용 메인 + (1 − s_t) × C_t
  s_t = 메인 북스톱 배율(이미 1일 지연, 인과). C 는 자체 볼타겟·스무딩·노출스케일을
  메인과 동일하게 적용한 독립 북(자체 스톱 없음) — 스톱으로 빈 만큼만 채운다.

후보 (사전등록 — 추가 금지, 다중검정 4개):
  C1 RV        밸류+캐리만 (둘 다 xs 중립) — 스톱은 방향성 손실로 걸리므로 상대가치는 살아있나
  C2 중립메인   4슬리브 전부 xs 중립화 — 메인 시그널의 국가간 성분만
  C3 커브       US2s10s·KR3s10s DV01 중립 스티프너 (동일국 동시마감 → 비동시 종가 아티팩트 없음)
  C4 바닥0.5   메인을 최소 절반은 유지 (s' = max(s, 0.5)) — 스톱 완화 대조군

사전등록 채택 기준 (모두 충족):
  (a) 스톱 구간에서 C 기여 손익 > 0 — 개발(2016–2021)·홀드아웃(2022+) 둘 다
  (b) 전체 북 ΔSR: 개발 ≥ +0.03 AND 홀드아웃 ≥ 0
  (c) T+2 ΔSR ≥ T+1 ΔSR × 0.7 (2016+)
  (d) 시간대 정직(한국 레그 T+2) ΔSR > 0 (2016+)
  (e) MaxDD 악화 ≤ 1.0%p (2016+)
불통과면 운용 변경 없음. 통과해도 config 토글 기본 OFF 로 두고 사용자 승인 후 켠다.

Usage: python scripts/test_stop_period_books.py
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
logging.disable(logging.WARNING)

from src.data.loader import DataLoader                            # noqa: E402
from src.sleeves.sleeve_engine import SleeveEngine, TRADING_DAYS  # noqa: E402
from scripts.run_sleeve_backtest import (                         # noqa: E402
    load_sleeve_config, cost_bps_for, DEFAULT_COSTS_BPS)

DEV = ('2016-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)
FULL = ('2016-01-01', None)
KR = ['KE1 Comdty', 'KAA1 Comdty']


def win(s, a, b):
    s = s[s.index >= pd.to_datetime(a)]
    return s[s.index <= pd.to_datetime(b)] if b else s


def sr(s):
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(TRADING_DAYS) if s.std() > 0 else 0.0


def mdd(s):
    eq = s.fillna(0).cumsum()
    return (eq - eq.cummax()).min()


def pnl(engine, pos, lag_map=None, gross_only=False):
    """futures 기준 일별 순손익 (비용 차감). lag_map: 자산별 보유 지연 (기본 1)."""
    R = [a for a in engine.rates_assets if a in pos.columns]
    costs = {**DEFAULT_COSTS_BPS, **(engine.cfg.get('costs_bps', {}) or {})}
    rets = engine.dir_returns[R].reindex(pos.index).fillna(0.0)
    held = pd.DataFrame({a: pos[a].shift((lag_map or {}).get(a, 1)) for a in R}).fillna(0.0)
    g = (held * rets).sum(axis=1)
    if gross_only:
        return g
    crate = pd.Series({a: cost_bps_for(a, costs) / 10000.0 for a in R})
    return g - pos[R].diff().abs().fillna(0.0).mul(crate, axis=1).sum(axis=1)


def main():
    ld = DataLoader()
    base = load_sleeve_config()
    px = ld.engine_prices(base)
    yl = ld.load_signal_yields(start_date='2010-01-01')
    mc = ld.load_signal_macro(start_date='2010-01-01')
    nostop = {**base['book_stop'], 'enabled': False}

    def build(cfg):
        e = SleeveEngine(px, config=cfg, yields=yl, macro=mc)
        return e, e.finalize_positions(e.compute_target_positions())

    em, pm = build(base)
    s = em._book_stop_state['scale'].reindex(pm.index).fillna(1.0)
    R = [a for a in em.rates_assets if a in pm.columns]
    eu, pu = build({**base, 'book_stop': nostop})

    zero_w = {k: 0.0 for k in base['sleeve_weights']}
    cands = {}
    _, cands['C1 RV'] = build({**base, 'book_stop': nostop,
                               'sleeve_weights': {**zero_w, 'value': 1.0, 'carry': 1.0}})
    _, cands['C2 중립메인'] = build({**base, 'book_stop': nostop,
                                  'xs_neutralize': {'trend': 1.0, 'value': 1.0, 'carry': 1.0,
                                                    'policy': 1.0}})
    _, cands['C3 커브'] = build({**base, 'book_stop': nostop, 'sleeve_weights': zero_w,
                               'curve_trades': {**base['curve_trades'], 'enabled': True}})

    stop = s < 1.0
    books = {'기준(현행)': pm[R]}
    for k, c in cands.items():
        books[k] = pm[R] + c[R].mul(1.0 - s, axis=0)
    books['C4 바닥0.5'] = pu[R].mul(np.maximum(s, 0.5), axis=0)

    lag_tz = {a: (2 if a in KR else 1) for a in R}
    res = {}
    for k, P in books.items():
        p1, p2, ptz = pnl(em, P), pnl(em, P, {a: 2 for a in R}), pnl(em, P, lag_tz)
        extra = (P - pm[R])                       # 기준 대비 추가 포지션 = 스톱 구간 대체 북
        contrib = pnl(em, extra, gross_only=False) if k != '기준(현행)' else p1 * 0
        res[k] = dict(
            sr_full=sr(win(p1, *FULL)), sr_dev=sr(win(p1, *DEV)), sr_hold=sr(win(p1, *HOLD)),
            sr_t2=sr(win(p2, *FULL)), sr_tz=sr(win(ptz, *FULL)), mdd=mdd(win(p1, *FULL)),
            ret=win(p1, *FULL).mean() * TRADING_DAYS,
            stop_dev=win(contrib[stop], *DEV).sum(), stop_hold=win(contrib[stop], *HOLD).sum())

    b = res['기준(현행)']
    rows = []
    for k, r in res.items():
        d = {'SR 16+': r['sr_full'], 'SR 개발': r['sr_dev'], 'SR 홀드': r['sr_hold'],
             'SR T+2': r['sr_t2'], 'SR 시간대정직': r['sr_tz'], '연수익%': r['ret'] * 100,
             'MaxDD%p': r['mdd'] * 100, '스톱구간기여 개발%': r['stop_dev'] * 100,
             '스톱구간기여 홀드%': r['stop_hold'] * 100}
        if k != '기준(현행)':
            dT1 = r['sr_full'] - b['sr_full']
            ok = {'a': r['stop_dev'] > 0 and r['stop_hold'] > 0,
                  'b': (r['sr_dev'] - b['sr_dev'] >= 0.03) and (r['sr_hold'] - b['sr_hold'] >= 0),
                  'c': (r['sr_t2'] - b['sr_t2']) >= 0.7 * dT1 if dT1 > 0 else False,
                  'd': r['sr_tz'] - b['sr_tz'] > 0,
                  'e': (r['mdd'] - b['mdd']) * 100 >= -1.0}
            d['판정'] = ('채택후보' if all(ok.values()) else
                       '불통과(' + ','.join(x for x, v in ok.items() if not v) + ')')
        else:
            d['판정'] = '-'
        rows.append(pd.Series(d, name=k))
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(f"스톱 구간 2016+: {int(win(stop, *FULL).sum())}일 / {len(win(stop, *FULL))}일")
    print(df.to_string(float_format=lambda x: f"{x:6.2f}"))

    print("\n후보 북 단독(상시 가동, 스톱 무관) SR — 참고:")
    for k, c in cands.items():
        p = pnl(em, c[R])
        print(f"   {k:<8} 16+ {sr(win(p, *FULL)):5.2f} | 개발 {sr(win(p, *DEV)):5.2f} | "
              f"홀드 {sr(win(p, *HOLD)):5.2f} | 메인과 상관 {win(p, *FULL).corr(win(pnl(em, pu[R]), *FULL)):+.2f}")

    print("\n연도별 스톱 구간 기여 (%):")
    yr = {}
    for k in books:
        if k == '기준(현행)':
            continue
        c = pnl(em, books[k] - pm[R])[stop]
        c = win(c, *FULL)
        yr[k] = (c.groupby(c.index.year).sum() * 100).round(2)
    print(pd.DataFrame(yr).to_string())


if __name__ == '__main__':
    main()
