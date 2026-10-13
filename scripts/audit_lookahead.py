# -*- coding: utf-8 -*-
"""금리 북 치팅 감사 — 비동시 종가 · 스톱 래치 · 비용 · 불변식 · 절단(미래참조) 테스트.

배경 (2026-07-22): 리버전 서브북이 SR 1.38 을 만들고 있었는데, 전부 '아시아 종가가 미국
종가보다 ~14시간 이르다'는 사실을 수확한 아티팩트였다. 같은 시기 book_stop 은 전기간 고점
래치로 느린 북을 3년간 꺼놓고 있었다. 2026-10-02/13 감사로 불변식(⑥)과 절단 테스트(⑦)를
추가했다. 새 시그널을 넣거나 파라미터를 바꾼 뒤에는 항상 이걸 돌릴 것.

판정 기준:
  · T+2 SR 이 T+1 대비 크게 무너지면 → 실행 타이밍에 기댄 가짜 알파 의심
  · 손익이 아시아/영국에 쏠리고 미국이 마이너스면 → 캐치업 수확 의심
  · 스톱 플랫 비율이 높고 최근 몇 년 내내 0 이면 → 래치 (dd_window 확인)
  · ⑥ 불변식 FAIL 또는 ⑦ 절단 차이 > 1e-9 → 성과와 무관하게 계산 결함 (종료코드 1)

엔진은 라이브와 같은 전체 패널로 돌리고 평가만 --start-date 부터 한다 (콜드스타트 왜곡 방지).

Usage: python scripts/audit_lookahead.py [--start-date 2016-01-01] [--trunc-dates 18]
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader                                   # noqa: E402
from src.sleeves.sleeve_engine import SleeveEngine, TRADING_DAYS         # noqa: E402
from src.sleeves.accounting import asset_pnl, perf_stats, window        # noqa: E402
from scripts.run_sleeve_backtest import load_sleeve_config, freshness_watch  # noqa: E402
from scripts.invariants import run_invariants, print_invariants, truncation_test  # noqa: E402

# 미국(시카고)보다 먼저 마감하는 시장 — 당일 미국 종가를 볼 수 없다.
EARLY_CLOSE = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
               'YM1 Comdty', 'XM1 Comdty', 'DU1 Comdty', 'RX1 Comdty',
               'OAT1 Comdty', 'IK1 Comdty']
US_BENCH = 'TY1 Comdty'


def main():
    p = argparse.ArgumentParser(description="금리 북 시차/래치/불변식/절단 감사")
    p.add_argument('--start-date', default='2016-01-01')
    p.add_argument('--trunc-dates', type=int, default=18)
    args = p.parse_args()
    start = args.start_date

    ld = DataLoader()
    cfg = load_sleeve_config()
    px, yl, mc = ld.engine_inputs(cfg)
    eng = SleeveEngine(px, config=cfg, yields=yl, macro=mc)
    R = eng.rates_assets
    pos = eng.finalize_positions(eng.compute_target_positions())
    early = [a for a in R if a in EARLY_CLOSE]

    def pnl(lag_map, cost_mult=1.0):
        return window(asset_pnl(eng, pos, R, lag_map=lag_map, cost_mult=cost_mult), start)

    print("=" * 84)
    print(f"  금리 북 치팅 감사  평가 {start} → {pos.index[-1].date()} (엔진 패널 {pos.index[0].date()}~)")
    _so = [a for a in R if a in eng.signal_only_assets]
    print(f"  시그널 {len(R)}종 / 매매 {len(R) - len(_so)}종"
          + (f" (시그널 전용: {', '.join(_so)})" if _so else ""))
    print("=" * 84)

    print("\n① 실행 시차 민감도 — 하루 늦게 집행해도 알파가 남는가")
    base = None
    for lag in (1, 2, 3):
        st = perf_stats(pnl({a: lag for a in R}).sum(axis=1))
        base = st['sharpe'] if lag == 1 else base
        print(f"   T+{lag}  SR {st['sharpe']:5.2f} | AnnRet {st['ret']:6.2%} | MaxDD {st['maxdd']:6.1%}")
    st_tz = perf_stats(pnl({a: (2 if a in EARLY_CLOSE else 1) for a in R}).sum(axis=1))
    print(f"   시간대 정직(조기마감 T+2)  SR {st_tz['sharpe']:5.2f} | AnnRet {st_tz['ret']:6.2%}")
    if base and base > 0.3 and st_tz['sharpe'] < base * 0.6:
        print("   ⚠ 시간대 정직 SR 이 40% 이상 무너짐 — 스테일 프라이스 수확 의심")

    print("\n② 손익 지역 쏠림 — 조기마감 시장이 전부 벌고 미국이 잃는가")
    net = pnl({a: 1 for a in R})
    tot = net.sum().sum()
    e_share = net[early].sum().sum() / tot if tot else float('nan')
    for a in sorted(R, key=lambda x: -net[x].sum()):
        tag = '조기마감' if a in EARLY_CLOSE else '미국(최종마감)'
        print(f"   {a:<14}{net[a].sum():>9.2%}{(net[a].sum() / tot if tot else 0):>7.0%}   {tag}")
    print(f"   조기마감 합계 비중 {e_share:.0%}")
    if e_share > 0.9:
        print("   ⚠ 조기마감 시장이 손익의 90% 초과 — 캐치업 수확 의심")

    print("\n③ 리드-래그 상관 — 각국은 미국의 '어제'를 따라가는가")
    us = eng.dir_returns[US_BENCH]
    print(f"   {'자산':<14}{'vs 美당일':>10}{'vs 美전일':>10}")
    for a in R:
        if a != US_BENCH:
            print(f"   {a:<14}{eng.dir_returns[a].corr(us):>10.3f}{eng.dir_returns[a].corr(us.shift(1)):>10.3f}")
    print("   (전일 상관 > 당일 상관이면 그 시장 가격은 구조적으로 하루 늦다)")

    print("\n④ book_stop 래치 점검")
    if not eng.book_stop_enabled:
        print("   스톱 OFF — 해당 없음")
    else:
        flat = window(pos[R].abs().sum(axis=1) < 1e-9, start)
        by_year = flat.groupby(flat.index.year).mean()
        print(f"   dd_window = {eng.book_stop_dd_window or '0 (전기간 ⚠래치 위험)'} | "
              f"임계 {eng.book_stop_dd_half:g}/{eng.book_stop_dd_flat:g}% | 전체 플랫 비율 {flat.mean():.0%}")
        print("   " + "  ".join(f"{y}:{v:.0%}" for y, v in by_year.items()))
        tail = by_year.tail(3)
        if len(tail) == 3 and (tail > 0.95).all():
            print("   ⚠ 최근 3년 연속 거의 완전 플랫 — 스톱이 래치됐을 가능성")

    turn = window(pos[R].diff().abs().fillna(0.0).sum(axis=1), start).mean() * TRADING_DAYS
    print(f"\n⑤ 비용 민감도 (회전 {turn:.0f}x/yr)")
    for mult, lbl in [(1, '가정대로'), (2, '2배'), (3, '3배')]:
        print(f"   비용 {lbl:<6} SR {perf_stats(pnl({a: 1 for a in R}, mult).sum(axis=1))['sharpe']:5.2f}")

    print("\n⑥ 불변식 점검 (롤 보정 · 팩터 퇴화 · DV01 상식 · 데이터 신선도 · 포지션)")
    n_fail = print_invariants(run_invariants(eng, ld, pos, watch=freshness_watch(eng)))

    print(f"\n⑦ 절단 테스트 — T 까지 자른 입력으로 처음부터 돌린 T일 결과 = 전체 표본 T일 결과? "
          f"({args.trunc_dates}개 날짜)")
    worst, rows = truncation_test(ld, cfg, pos, n_dates=args.trunc_dates, start=start)
    for T, d in rows:
        print(f"   T={T.date()}  최대 차이 {d:.1e}")
    print(f"   → 최악 {worst:.1e} {'✅ 미래참조 없음' if worst < 1e-9 else '❌ 미래참조/비인과 계산 의심'}")
    return 1 if (n_fail or worst >= 1e-9) else 0


if __name__ == '__main__':
    sys.exit(main())
