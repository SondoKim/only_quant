# -*- coding: utf-8 -*-
"""이전 vs 현재 금리 북 백테스트 비교 PNG (2026-10-02 로직 감사 전후).

이전 = roll_adjust OFF · pnl_basis yield_implied · carry legacy (감사 전 운용 로직)
현재 = config/indicators.yaml 그대로 (롤 보정 · 실선물 손익 · hedged_xs 캐리)
둘 다 운용 백테스트와 같은 방식(run_sleeve_backtest.run, 2016-01-01 시작, 비용 차감 net).

패널: ① 누적 손익(배정자본 대비 %, 단순합 — 대시보드 YTD 와 같은 방식)
      ② 드로다운  ③ 1년 롤링 SR  ④ 연도별 수익률

Usage: python scripts/plot_backtest_compare.py [--start-date 2016-01-01] [--out PATH]
"""
import sys
import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from scripts.run_sleeve_backtest import run, TRADING_DAYS  # noqa: E402

OLD_CFG = {'roll_adjust': False, 'pnl_basis': 'yield_implied', 'carry_method': 'legacy'}

# dataviz 기본 팔레트 categorical 1·2 (validate_palette.js light 통과)
C_NEW, C_OLD = '#2a78d6', '#eb6834'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'


def stats(s):
    sr = s.mean() / s.std() * np.sqrt(TRADING_DAYS) if s.std() > 0 else 0.0
    eq = s.cumsum()
    dd = eq - eq.cummax()
    return sr, s.mean() * TRADING_DAYS, dd.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-date', default='2016-01-01')
    ap.add_argument('--out', default=str(ROOT / 'reports' / 'backtest_compare' /
                                         f'rates_backtest_old_vs_new_{date.today().isoformat()}.png'))
    args = ap.parse_args()

    old = run(start_date=args.start_date, plot=False, save_outputs=False, cfg_override=OLD_CFG)['rates']
    new = run(start_date=args.start_date, plot=False, save_outputs=False)['rates']
    idx = old.index.intersection(new.index)
    old, new = old.reindex(idx).fillna(0.0), new.reindex(idx).fillna(0.0)

    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig = plt.figure(figsize=(13, 13), facecolor=SURF)
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.3, 1.3, 1.6], hspace=0.42)
    ax = [fig.add_subplot(gs[i]) for i in range(4)]
    for a in ax:
        a.set_facecolor(SURF)
        a.grid(axis='y', color=GRID, lw=0.8)
        a.spines[['top', 'right']].set_visible(False)

    so, ao, mo = stats(old)
    sn, an, mn = stats(new)
    lab_old = f"이전 (감사 전)  SR {so:.2f} · 연 {ao:.2%} · MaxDD {mo:.2%}"
    lab_new = f"현재 (롤보정·실선물손익·캐리교정)  SR {sn:.2f} · 연 {an:.2%} · MaxDD {mn:.2%}"

    # ① 누적
    co, cn = old.cumsum() * 100, new.cumsum() * 100
    ax[0].plot(co.index, co, color=C_OLD, lw=2, label=lab_old)
    ax[0].plot(cn.index, cn, color=C_NEW, lw=2, label=lab_new)
    for c, col, nm in [(co, C_OLD, '이전'), (cn, C_NEW, '현재')]:
        ax[0].annotate(f"{nm} {c.iloc[-1]:+.1f}%", (c.index[-1], c.iloc[-1]),
                       xytext=(6, 0), textcoords='offset points', va='center',
                       fontsize=10, color=INK, fontweight='bold')
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set_title(f"금리 북 백테스트 — 이전 vs 현재 로직 ({idx[0].date()} → {idx[-1].date()}, "
                    f"비용 차감 net, 배정자본 대비)", loc='left', fontsize=13, fontweight='bold')
    ax[0].set_ylabel('누적 손익 (%, 단순합)')
    ax[0].legend(loc='upper left', frameon=False, fontsize=10)
    ax[0].margins(x=0.06)

    # ② 드로다운
    for s, col, nm in [(old, C_OLD, '이전'), (new, C_NEW, '현재')]:
        eq = s.cumsum() * 100
        ax[1].plot(eq.index, eq - eq.cummax(), color=col, lw=1.6, label=nm)
    ax[1].set_ylabel('드로다운 (%p)')
    ax[1].set_title('드로다운', loc='left', fontsize=11)
    ax[1].legend(loc='lower left', frameon=False, fontsize=9, ncol=2)
    ax[1].margins(x=0.06)

    # ③ 1년 롤링 SR
    for s, col, nm in [(old, C_OLD, '이전'), (new, C_NEW, '현재')]:
        r = s.rolling(TRADING_DAYS, min_periods=200)
        ax[2].plot(s.index, r.mean() / r.std() * np.sqrt(TRADING_DAYS), color=col, lw=1.6, label=nm)
    ax[2].axhline(0, color=INK2, lw=0.8)
    ax[2].set_ylabel('SR')
    ax[2].set_title('1년 롤링 샤프', loc='left', fontsize=11)
    ax[2].legend(loc='lower left', frameon=False, fontsize=9, ncol=2)
    ax[2].set_xlim(ax[0].get_xlim())   # 롤링 워밍업 때문에 시작이 늦어도 위 패널과 시간축 정렬

    # ④ 연도별 수익률
    yo = old.groupby(old.index.year).sum() * 100
    yn = new.groupby(new.index.year).sum() * 100
    x = np.arange(len(yo))
    w = 0.38
    ax[3].bar(x - w / 2 - 0.01, yo.values, w, color=C_OLD, label='이전')
    ax[3].bar(x + w / 2 + 0.01, yn.values, w, color=C_NEW, label='현재')
    for i, (vo, vn) in enumerate(zip(yo.values, yn.values)):
        for xv, v in ((i - w / 2 - 0.01, vo), (i + w / 2 + 0.01, vn)):
            ax[3].text(xv, v + (0.12 if v >= 0 else -0.12), f"{v:.1f}", ha='center',
                       va='bottom' if v >= 0 else 'top', fontsize=8, color=INK2)
    ax[3].axhline(0, color=INK2, lw=0.8)
    ax[3].set_xticks(x, [str(y) + ('(YTD)' if y == idx[-1].year else '') for y in yo.index])
    ax[3].set_ylabel('연 수익률 (%)')
    ax[3].set_title('연도별 수익률', loc='left', fontsize=11)
    ax[3].legend(loc='upper left', frameon=False, fontsize=9, ncol=2)

    fig.text(0.01, 0.005,
             "이전 = roll_adjust OFF · pnl_basis yield_implied · carry legacy   |   "
             "현재 = config/indicators.yaml (roll_adjust · futures 손익 · hedged_xs 캐리)   |   "
             "scripts/plot_backtest_compare.py", fontsize=8.5, color=INK2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor=SURF, bbox_inches='tight')
    print(f"\n✅ 저장: {out}")
    print(f"   이전 SR {so:.2f} / 연 {ao:.2%} / MaxDD {mo:.2%}")
    print(f"   현재 SR {sn:.2f} / 연 {an:.2%} / MaxDD {mn:.2%}")


if __name__ == '__main__':
    main()
