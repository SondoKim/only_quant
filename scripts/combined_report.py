"""
Combined-book report: factory FX book + sleeve rates book.

Reads the two backtest logs (each produced by its own engine under the honest
next-bar convention, net of costs), normalizes both daily PnL streams to a
common volatility target, combines them at the given risk weights, and reports
per-stream / combined Sharpe, correlation, drawdown and an equity-curve plot.

Inputs (run these first if missing/stale):
    trading_log.csv          — python scripts/run_backtest.py --skip-discovery
                               (factory FX book; rates are no longer in the factory)
    sleeve_backtest_log.csv  — python scripts/run_sleeve_backtest.py
                               (sleeve rates book; 'rates' column = daily net return)

Usage:
    python scripts/combined_report.py
    python scripts/combined_report.py --fx-weight 0.5 --rates-weight 0.5 --target-vol 0.10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def ann_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(TRADING_DAYS))


def max_dd(r: pd.Series) -> float:
    """Max drawdown of an additive PnL stream (in the stream's own units)."""
    eq = r.fillna(0.0).cumsum()
    return float((eq - eq.cummax()).min())


def load_streams(root: Path):
    f_path = root / 'trading_log.csv'
    s_path = root / 'sleeve_backtest_log.csv'
    if not f_path.exists():
        raise SystemExit("❌ trading_log.csv 없음 — 먼저 run_backtest.py 를 실행하세요.")
    if not s_path.exists():
        raise SystemExit("❌ sleeve_backtest_log.csv 없음 — 먼저 run_sleeve_backtest.py 를 실행하세요.")

    fac = pd.read_csv(f_path, parse_dates=['Date']).set_index('Date')
    slv = pd.read_csv(s_path, index_col=0, parse_dates=True)

    # Factory FX book: cumulative PnL in %-points → daily diff
    fx = fac['total_fx_cumpnl'].diff().rename('fx_factory')
    # Sleeve rates book: daily net return (decimal) → %-points for same units
    rt = (slv['rates'] * 100.0).rename('rates_sleeve')
    return fx, rt


def main():
    ap = argparse.ArgumentParser(description="통합 북 리포트 (공장 FX + 슬리브 금리)")
    ap.add_argument('--fx-weight', type=float, default=0.5,
                    help='FX 북 리스크 가중 (기본 0.5 = 동일 리스크)')
    ap.add_argument('--rates-weight', type=float, default=0.5,
                    help='금리 북 리스크 가중 (기본 0.5)')
    ap.add_argument('--target-vol', type=float, default=0.10,
                    help='각 스트림을 정규화할 연 변동성 (기본 10%%)')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    fx, rt = load_streams(root)

    # Align on common dates (inner join: both books live)
    df = pd.concat([fx, rt], axis=1, join='inner').dropna(how='all').fillna(0.0)
    if len(df) < 60:
        raise SystemExit(f"❌ 두 로그의 공통 구간이 너무 짧습니다 ({len(df)}일). "
                         f"같은 기간으로 두 백테스트를 다시 돌리세요.")
    start, end = df.index[0].date(), df.index[-1].date()

    # Normalize each stream to target vol (full-sample scale — reporting only;
    # live sizing is each engine's own vol targeting)
    tv_daily = args.target_vol / np.sqrt(TRADING_DAYS) * 100.0   # %-points/day
    norm = {}
    for col in df.columns:
        sd = df[col].std()
        norm[col] = df[col] * (tv_daily / sd) if sd > 0 else df[col]
    n = pd.DataFrame(norm)

    wsum = args.fx_weight + args.rates_weight
    combined = (n['fx_factory'] * args.fx_weight + n['rates_sleeve'] * args.rates_weight) / wsum
    corr = float(n['fx_factory'].corr(n['rates_sleeve']))

    print("\n" + "=" * 66)
    print(f"  통합 북 리포트  {start} → {end}  ({len(df)}일)")
    print(f"  리스크 가중: FX {args.fx_weight:g} / 금리 {args.rates_weight:g} · "
          f"스트림 정규화 {args.target_vol:.0%} vol")
    print("=" * 66)
    rows = [('FX (전략 공장)', n['fx_factory']),
            ('금리 (슬리브)', n['rates_sleeve']),
            ('통합 북', combined)]
    print(f"  {'북':<14} {'Sharpe':>7} {'연변동성':>8} {'MaxDD':>8}")
    for name, s in rows:
        print(f"  {name:<14} {ann_sharpe(s):>7.2f} {ann_vol(s):>7.1f}% {max_dd(s):>7.1f}%")
    print(f"\n  두 스트림 일별 상관: {corr:+.2f}"
          + ("  → 분산효과 큼" if corr < 0.2 else ""))

    yr = combined.groupby(combined.index.year).apply(ann_sharpe)
    print("\n  연도별 통합 Sharpe: "
          + "  ".join(f"{y} {v:+.2f}" for y, v in yr.items()))

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 7))
        for name, s in rows:
            label_en = {'FX (전략 공장)': 'FX (factory)',
                        '금리 (슬리브)': 'Rates (sleeve)',
                        '통합 북': 'COMBINED'}[name]
            lw = 2.6 if name == '통합 북' else 1.4
            ax.plot(s.index, s.cumsum(), lw=lw,
                    label=f"{label_en} (SR {ann_sharpe(s):.2f})")
        ax.set_title(f"Combined Book — factory FX + sleeve rates  "
                     f"({start} → {end}, corr {corr:+.2f})")
        ax.set_ylabel("Cumulative PnL (%-pts, vol-normalized)")
        ax.grid(alpha=0.3)
        ax.legend()
        out = f"combined_book_{start}_to{end}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\n📊 통합 에쿼티 커브 저장 → {out}")


if __name__ == '__main__':
    main()
