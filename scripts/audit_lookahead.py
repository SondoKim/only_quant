# -*- coding: utf-8 -*-
"""금리 북 치팅 감사 — 비동시 종가(스테일 프라이스) & 스톱 래치 상시 점검.

배경 (2026-07-22): 리버전 서브북이 SR 1.38 을 만들고 있었는데, 전부 '아시아
종가가 미국 종가보다 ~14시간 이르다'는 사실을 수확한 아티팩트였다. 하루만
늦춰 집행하면 SR 1.01 → -0.14 로 소멸했고, 미국 종가를 보고 한국 종가에
체결하는 것은 불가능하다. 같은 시기 book_stop 은 전기간 고점 래치로 느린 북을
3년간 완전히 꺼놓고 있었다.

이 자산군(美·英·日·韓)에서 비동시 종가는 구조적이므로, 횡단면 중립 + 빠른
시그널 조합은 언제든 같은 아티팩트를 재생산한다. 새 시그널을 넣거나 파라미터를
바꾼 뒤에는 항상 이걸 돌릴 것.

판정 기준:
  · T+2 SR 이 T+1 대비 크게 무너지면 → 실행 타이밍에 기댄 가짜 알파 의심
  · 손익이 아시아/영국에 쏠리고 미국이 마이너스면 → 캐치업 수확 의심
  · 스톱 플랫 비율이 높고 최근 몇 년 내내 0 이면 → 래치 (dd_window 확인)

Usage: python scripts/audit_lookahead.py [--start-date 2016-01-01]
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader                       # noqa: E402
from src.data.preprocessor import DataPreprocessor           # noqa: E402
from src.sleeves.sleeve_engine import SleeveEngine, TRADING_DAYS  # noqa: E402
from scripts.run_sleeve_backtest import (                    # noqa: E402
    load_sleeve_config, perf_stats, cost_bps_for, DEFAULT_COSTS_BPS)

# 미국(시카고)보다 먼저 마감하는 시장 — 당일 미국 종가를 볼 수 없다.
EARLY_CLOSE = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
               'YM1 Comdty', 'XM1 Comdty', 'DU1 Comdty', 'RX1 Comdty',
               'OAT1 Comdty', 'IK1 Comdty']
US_BENCH = 'TY1 Comdty'


def _net_pnl(engine, pos, lag_map):
    """금리환산 수익률 기준 일별 순손익 (run_sleeve_backtest 와 동일 방식)."""
    R = [a for a in engine.rates_assets if a in pos.columns]
    costs = {**DEFAULT_COSTS_BPS, **(engine.cfg.get('costs_bps', {}) or {})}
    dirr = engine.dir_returns[R].reindex(pos.index).fillna(0.0)
    rets = dirr.copy()
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
    held = pd.DataFrame({a: pos[a].shift(lag_map.get(a, 1)) for a in R}).fillna(0.0)
    return held * rets[R] - pos[R].diff().abs().fillna(0.0).mul(crate, axis=1)


def main():
    p = argparse.ArgumentParser(description="금리 북 시차/래치 감사")
    p.add_argument('--start-date', default='2016-01-01')
    args = p.parse_args()

    ld = DataLoader()
    px = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    yl = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    px = px[px.index >= pd.to_datetime(args.start_date)]
    cfg = load_sleeve_config()

    eng = SleeveEngine(px, config=cfg, yields=yl)
    R = eng.rates_assets
    pos = eng.finalize_positions(eng.compute_target_positions())
    early = [a for a in R if a in EARLY_CLOSE]

    print("=" * 84)
    print(f"  금리 북 치팅 감사  {pos.index[0].date()} → {pos.index[-1].date()}")
    _so = [a for a in R if a in eng.signal_only_assets]
    print(f"  시그널 {len(R)}종 / 매매 {len(R) - len(_so)}종"
          + (f" (시그널 전용: {', '.join(_so)})" if _so else "")
          + f" | 리버전 {'ON ⚠' if eng.reversion_enabled else 'OFF'}")
    print("=" * 84)

    print("\n① 실행 시차 민감도 — 하루 늦게 집행해도 알파가 남는가")
    base = None
    for lag in (1, 2, 3):
        s = _net_pnl(eng, pos, {a: lag for a in R}).sum(axis=1).dropna()
        st = perf_stats(s)
        if lag == 1:
            base = st['sharpe']
        print(f"   T+{lag}  SR {st['sharpe']:5.2f} | AnnRet {st['ret']:6.2%} "
              f"| MaxDD {st['maxdd']:6.1%}")
    tz = {a: (2 if a in EARLY_CLOSE else 1) for a in R}
    s_tz = _net_pnl(eng, pos, tz).sum(axis=1).dropna()
    st_tz = perf_stats(s_tz)
    print(f"   시간대 정직(조기마감 T+2)  SR {st_tz['sharpe']:5.2f} "
          f"| AnnRet {st_tz['ret']:6.2%}")
    if base and base > 0.3 and st_tz['sharpe'] < base * 0.6:
        print("   ⚠ 시간대 정직 SR 이 40% 이상 무너짐 — 스테일 프라이스 수확 의심")

    print("\n② 손익 지역 쏠림 — 조기마감 시장이 전부 벌고 미국이 잃는가")
    net = _net_pnl(eng, pos, {a: 1 for a in R})
    tot = net.sum().sum()
    e_share = net[early].sum().sum() / tot if tot else float('nan')
    for a in sorted(R, key=lambda x: -net[x].sum()):
        tag = '조기마감' if a in EARLY_CLOSE else '미국(최종마감)'
        print(f"   {a:<14}{net[a].sum():>9.2%}{net[a].sum()/tot:>7.0%}   {tag}")
    print(f"   조기마감 합계 비중 {e_share:.0%}")
    if e_share > 0.9:
        print("   ⚠ 조기마감 시장이 손익의 90% 초과 — 캐치업 수확 의심")

    print("\n③ 리드-래그 상관 — 각국은 미국의 '어제'를 따라가는가")
    us = eng.dir_returns[US_BENCH]
    print(f"   {'자산':<14}{'vs 美당일':>10}{'vs 美전일':>10}")
    for a in R:
        if a == US_BENCH:
            continue
        print(f"   {a:<14}{eng.dir_returns[a].corr(us):>10.3f}"
              f"{eng.dir_returns[a].corr(us.shift(1)):>10.3f}")
    print("   (전일 상관 > 당일 상관이면 그 시장 가격은 구조적으로 하루 늦다)")

    print("\n④ book_stop 래치 점검")
    if not eng.book_stop_enabled:
        print("   스톱 OFF — 해당 없음")
    else:
        flat = (pos[R].abs().sum(axis=1) < 1e-9)
        by_year = flat.groupby(flat.index.year).mean()
        print(f"   dd_window = "
              f"{eng.book_stop_dd_window or '0 (전기간 ⚠래치 위험)'} | "
              f"전체 플랫 비율 {flat.mean():.0%}")
        print("   " + "  ".join(f"{y}:{v:.0%}" for y, v in by_year.items()))
        tail = by_year.tail(3)
        if len(tail) == 3 and (tail > 0.95).all():
            print("   ⚠ 최근 3년 연속 거의 완전 플랫 — 스톱이 래치됐을 가능성")

    print("\n⑤ 비용 민감도 (회전 "
          f"{pos[R].diff().abs().fillna(0.0).sum(axis=1).mean()*TRADING_DAYS:.0f}x/yr)")
    for mult, lbl in [(1, '가정대로'), (2, '2배'), (3, '3배')]:
        n = _net_pnl(eng, pos, {a: 1 for a in R})
        turn = pos[R].diff().abs().fillna(0.0)
        costs = {**DEFAULT_COSTS_BPS, **(cfg.get('costs_bps', {}) or {})}
        crate = pd.Series({a: cost_bps_for(a, costs) / 10000.0 for a in R})
        s = (n + turn.mul(crate, axis=1) - turn.mul(crate, axis=1) * mult).sum(axis=1)
        print(f"   비용 {lbl:<6} SR {perf_stats(s.dropna())['sharpe']:5.2f}")


if __name__ == '__main__':
    main()
