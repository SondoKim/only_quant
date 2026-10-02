# -*- coding: utf-8 -*-
"""롤 보정 + 캐리 산식 A/B (2026-10-02 실매매 로직 감사 후속).

감사 결과 두 가지:
  ① 금리 연속선물(XX1 Comdty)이 무보정 — 롤일마다 가짜 수익률(KE1 잔차 7.8σ,
     TY1 평균 −22bp)이 트렌드·실현볼·볼타겟·북스톱 섀도우 손익에 들어갔다.
     손익만 금리환산(yield_implied)으로 우회해 왔다.
  ② rates carry(legacy) 가 KE 에서 US 2Y 와 비트 동일 — 테너금리=펀딩금리라
     자국 정보가 0. 자산별 시계열 z 가 국가간 레벨 비교를 지운다.

변형 (모두 같은 패널·같은 평가 구간):
  V0 현행           roll OFF · pnl yield_implied · carry legacy
  V1 롤보정(시그널)  roll ON  · pnl yield_implied · carry legacy   ← 시그널 변화만 분리
  V2 롤보정+실손익   roll ON  · pnl futures       · carry legacy
  V3 + 캐리 교정     roll ON  · pnl futures       · carry hedged_xs
  V4 캐리 OFF 대조   roll ON  · pnl futures       · carry weight 0

사전등록 판정 (결과 보기 전 고정):
  · 롤 보정 + futures 손익 = 데이터 교정이므로 SR 과 무관하게 채택 (영향만 보고).
  · hedged_xs 채택 조건 (V3 vs V2): 개발(2016–2021) SR ≥ V2 − 0.05 AND
    홀드아웃(2022+) SR ≥ V2 − 0.05 AND V3 의 T+2 SR ≥ V3 T+1 − 0.10.
    불통과면 legacy 유지 + 대시보드 설명을 실제 계산에 맞게 정정, 사용자 보고.

Usage: python scripts/test_roll_carry_ab.py
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

VARIANTS = {
    'V0 현행':          dict(roll_adjust=False, pnl_basis='yield_implied', carry_method='legacy'),
    'V1 롤보정(시그널)': dict(roll_adjust=True, pnl_basis='yield_implied', carry_method='legacy'),
    'V2 롤보정+실손익':  dict(roll_adjust=True, pnl_basis='futures', carry_method='legacy'),
    'V3 +캐리교정':     dict(roll_adjust=True, pnl_basis='futures', carry_method='hedged_xs'),
    'V4 캐리OFF':       dict(roll_adjust=True, pnl_basis='futures', carry_method='legacy',
                           carry_off=True),
}


def sr(s):
    s = s.dropna()
    return s.mean() / s.std() * np.sqrt(TRADING_DAYS) if s.std() > 0 else 0.0


def mdd(s):
    eq = (1 + s.fillna(0)).cumprod()
    return (eq / eq.cummax() - 1).min()


def win(s, a, b):
    s = s[s.index >= pd.to_datetime(a)]
    return s[s.index <= pd.to_datetime(b)] if b else s


def book_pnl(engine, pos, basis, lag=1):
    R = [a for a in engine.rates_assets if a in pos.columns]
    costs = {**DEFAULT_COSTS_BPS, **(engine.cfg.get('costs_bps', {}) or {})}
    dirr = engine.dir_returns[R].reindex(pos.index).fillna(0.0)
    rets = dirr.copy()
    if basis != 'futures':
        for a in R:
            yt = engine.tradeable_yield_map.get(a)
            if yt is None or engine.yields is None or yt not in engine.yields.columns:
                continue
            dy = engine.yields[yt].reindex(dirr.index).diff() * 100.0
            beta = (dirr[a].rolling(250, min_periods=60).cov(dy)
                    / dy.rolling(250, min_periods=60).var()).shift(1)
            imp = beta * dy
            rets[a] = imp.where(imp.notna(), dirr[a])
    crate = pd.Series({a: cost_bps_for(a, costs) / 10000.0 for a in R})
    held = pos[R].shift(lag).fillna(0.0)
    return (held * rets[R]).sum(axis=1) - pos[R].diff().abs().fillna(0.0).mul(crate, axis=1).sum(axis=1)


def main():
    ld = DataLoader()
    base = load_sleeve_config()
    px_raw = ld.engine_prices({**base, 'roll_adjust': False})
    px_adj = ld.engine_prices({**base, 'roll_adjust': True})
    yl = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    mc = ld.load_signal_macro(start_date='2010-01-01', use_cache=True)
    print(f"패널 {px_raw.index[0].date()} → {px_raw.index[-1].date()} | 롤 보정 적용: "
          f"{len(ld.load_log.get('roll_adjust', {}).get('applied', []))}종, "
          f"건너뜀 {ld.load_log.get('roll_adjust', {}).get('skipped')}")

    rows, pos_last, stop_flat = [], {}, {}
    for name, v in VARIANTS.items():
        cfg = {**base, 'roll_adjust': v['roll_adjust'], 'pnl_basis': v['pnl_basis'],
               'carry_method': v['carry_method']}
        if v.get('carry_off'):
            cfg['sleeve_weights'] = {**base['sleeve_weights'], 'carry': 0.0}
        eng = SleeveEngine(px_adj if v['roll_adjust'] else px_raw, config=cfg, yields=yl, macro=mc)
        pos = eng.finalize_positions(eng.compute_target_positions())
        p1 = book_pnl(eng, pos, v['pnl_basis'], 1)
        p2 = book_pnl(eng, pos, v['pnl_basis'], 2)
        sc = eng._book_stop_state['scale']
        stop_flat[name] = {y: f"{(g == 0).mean():.0%}" for y, g in win(sc, *FULL).groupby(win(sc, *FULL).index.year)}
        T = [a for a in eng.rates_assets if a not in eng.signal_only_assets]
        pos_last[name] = pos[T].iloc[-1].round(3)
        rows.append({'변형': name,
                     'SR 2016+': sr(win(p1, *FULL)), 'SR 개발16-21': sr(win(p1, *DEV)),
                     'SR 홀드22+': sr(win(p1, *HOLD)), 'T+2 2016+': sr(win(p2, *FULL)),
                     'AnnVol': win(p1, *FULL).std() * np.sqrt(TRADING_DAYS),
                     'MaxDD': mdd(win(p1, *FULL)),
                     '2026YTD': win(p1, '2026-01-01', None).sum()})
    df = pd.DataFrame(rows).set_index('변형')
    pd.set_option('display.width', 200)
    print("\n" + df.to_string(float_format=lambda x: f"{x:6.3f}"))
    print("\n북스톱 플랫 비율 (연도별):")
    print(pd.DataFrame(stop_flat).T.to_string())
    print("\n최신 목표 포지션:")
    print(pd.DataFrame(pos_last).to_string())

    v2, v3 = df.loc['V2 롤보정+실손익'], df.loc['V3 +캐리교정']
    ok = (v3['SR 개발16-21'] >= v2['SR 개발16-21'] - 0.05
          and v3['SR 홀드22+'] >= v2['SR 홀드22+'] - 0.05
          and v3['T+2 2016+'] >= v3['SR 2016+'] - 0.10)
    print(f"\n사전등록 판정 — hedged_xs 캐리: {'채택' if ok else '불통과 (legacy 유지)'}")


if __name__ == '__main__':
    main()
