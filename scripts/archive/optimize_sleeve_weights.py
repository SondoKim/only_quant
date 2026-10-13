"""
Walk-forward sleeve-weight allocation.

Treats each sleeve (Trend / Value / Carry) as a standalone return stream
(a sub-portfolio sized to ~target vol), then allocates capital ACROSS sleeves
using ONLY trailing data, rebalanced monthly. Strictly out-of-sample: the
weights applied in month m use data up to the end of month m-1.

Schemes compared (all reported — no cherry-picking):
  - individual sleeves (reference)
  - equal           : 1/N  (the benchmark to beat)
  - risk_parity     : w ∝ 1/trailing_vol  (equal risk, no return-chasing)
  - sharpe_shrunk   : w ∝ max(0, trailing_Sharpe) + shrink  (adaptive, regularized)

Low degrees of freedom (3 sleeve weights, computed from trailing stats) keeps
this far from the data-mining trap of tuning signal parameters to the sample.

Usage:
    python scripts/optimize_sleeve_weights.py --start-date 2020-01-01
    python scripts/optimize_sleeve_weights.py --start-date 2020-01-01 --by-class
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.sleeves.sleeve_engine import SleeveEngine, TRADING_DAYS
from scripts.run_sleeve_backtest import (
    load_sleeve_config, cost_bps_for, perf_stats, DEFAULT_COSTS_BPS,
)

SLEEVES = ['trend', 'value', 'carry']
TRAIL_WINDOW = 252          # trailing window for weight estimation
MIN_HISTORY = 252           # need this much before adapting (else equal weight)
SHARPE_SHRINK = 0.25        # added to clipped Sharpe so weights never go to 0


def sleeve_net_returns(prices, yields, cfg_base, costs, by_class):
    """Daily net-return stream for each sleeve (optionally split by asset class).

    Returns a DataFrame: columns = sleeve (or f'{cls}:{sleeve}'), rows = dates.
    """
    out = {}
    for sleeve in SLEEVES:
        cfg = dict(cfg_base)
        cfg['sleeve_weights'] = {s: (1.0 if s == sleeve else 0.0) for s in SLEEVES}
        # Regime gate only affects Trend; skip its cost for value/carry runs.
        if sleeve != 'trend':
            rg = dict(cfg.get('regime_gate', {}) or {})
            rg['enabled'] = False
            cfg['regime_gate'] = rg
        eng = SleeveEngine(prices, config=cfg, yields=yields)
        pos = eng.compute_target_positions()
        if pos.empty:
            continue
        dr = eng.dir_returns[pos.columns].reindex(pos.index).fillna(0.0)
        gross = pos.shift(1).fillna(0.0) * dr
        turn = pos.diff().abs().fillna(0.0)
        crate = pd.Series({a: cost_bps_for(a, costs) / 1e4 for a in pos.columns})
        net = gross - turn.mul(crate, axis=1)
        if by_class:
            R = [c for c in pos.columns if 'Comdty' in c]
            F = [c for c in pos.columns if 'Curncy' in c]
            if R: out[f'rates:{sleeve}'] = net[R].sum(axis=1)
            if F: out[f'fx:{sleeve}'] = net[F].sum(axis=1)
        else:
            out[sleeve] = net.sum(axis=1)
    return pd.DataFrame(out).dropna()


# ── Allocation schemes: trailing stats → weights (sum to 1) ─────────────────
def w_equal(hist):
    n = hist.shape[1]
    return np.ones(n) / n


def w_risk_parity(hist):
    vol = hist.std().replace(0, np.nan)
    w = (1.0 / vol).fillna(0.0).values
    return w / w.sum() if w.sum() > 0 else w_equal(hist)


def w_sharpe_shrunk(hist):
    sr = (hist.mean() / hist.std().replace(0, np.nan) * np.sqrt(TRADING_DAYS)).fillna(0.0)
    sr = sr.clip(lower=0.0) + SHARPE_SHRINK
    w = sr.values
    return w / w.sum() if w.sum() > 0 else w_equal(hist)


SCHEMES = {'equal': w_equal, 'risk_parity': w_risk_parity, 'sharpe_shrunk': w_sharpe_shrunk}


def walk_forward_combine(streams: pd.DataFrame, scheme_fn) -> pd.Series:
    """Monthly-rebalanced OOS combination of sleeve streams."""
    month_ends = streams.resample('ME').last().index
    weights = pd.DataFrame(index=streams.index, columns=streams.columns, dtype=float)

    for me in month_ends:
        hist = streams.loc[streams.index <= me]
        if len(hist) >= MIN_HISTORY:
            w = scheme_fn(hist.iloc[-TRAIL_WINDOW:])
        else:
            w = w_equal(streams)
        # apply to the NEXT month (strictly OOS)
        future = streams.index[streams.index > me]
        if len(future):
            weights.loc[future[0]:, :] = w  # ffill-style: overwritten by later months

    weights = weights.ffill().fillna(1.0 / streams.shape[1])
    return (streams * weights).sum(axis=1)


def run(start_date=None, end_date=None, by_class=False, plot=True):
    print("📊 Loading data...")
    loader = DataLoader()
    prices = DataPreprocessor(loader.load_data(use_cache=True)).clean().get_data()
    if start_date:
        prices = prices[prices.index >= pd.to_datetime(start_date)]
    if end_date:
        prices = prices[prices.index <= pd.to_datetime(end_date)]
    yields = loader.load_signal_yields(use_cache=True)

    cfg = load_sleeve_config()
    costs = {**DEFAULT_COSTS_BPS, **(cfg.get('costs_bps', {}) or {})}

    print("   Computing per-sleeve return streams...")
    streams = sleeve_net_returns(prices, yields, cfg, costs, by_class)

    print("\n" + "=" * 70)
    print(f"  SLEEVE-WEIGHT ALLOCATION  {streams.index[0].date()} → {streams.index[-1].date()}"
          f"  {'(by class)' if by_class else ''}")
    print("=" * 70)
    print("  Individual sleeves (reference):")
    for col in streams.columns:
        s = perf_stats(streams[col])
        print(f"    {col:<16} Sharpe {s['sharpe']:5.2f} | Vol {s['vol']:5.1%} | MaxDD {s['maxdd']:6.1%}")

    print("\n  Walk-forward allocations (OOS):")
    results = {}
    for name, fn in SCHEMES.items():
        combo = walk_forward_combine(streams, fn)
        results[name] = combo
        s = perf_stats(combo)
        print(f"    {name:<16} Sharpe {s['sharpe']:5.2f} | Vol {s['vol']:5.1%} | "
              f"AnnRet {s['ret']:6.1%} | MaxDD {s['maxdd']:6.1%}")

    if plot:
        plt.figure(figsize=(13, 7))
        for name, combo in results.items():
            s = perf_stats(combo)
            (1 + combo).cumprod().plot(label=f"{name} (SR {s['sharpe']:.2f})", lw=1.6)
        for col in streams.columns:
            (1 + streams[col]).cumprod().plot(label=f"  {col}", lw=0.7, alpha=0.4)
        plt.title("Walk-forward sleeve allocation (OOS) vs individual sleeves")
        plt.ylabel("Cumulative (1+r)"); plt.grid(alpha=0.3); plt.legend(fontsize=8)
        sd = streams.index[0].strftime('%Y-%m-%d')
        out = f"sleeve_alloc_{sd}{'_byclass' if by_class else ''}.png"
        plt.tight_layout(); plt.savefig(out, dpi=110)
        print(f"\n📊 Saved plot → {out}")

    return streams, results


def main():
    p = argparse.ArgumentParser(description="Walk-forward sleeve-weight allocation")
    p.add_argument('--start-date', default='2020-01-01')
    p.add_argument('--end-date', default=None)
    p.add_argument('--by-class', action='store_true',
                   help='Allocate across class×sleeve (6 streams) instead of 3 sleeves')
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()
    run(start_date=args.start_date, end_date=args.end_date,
        by_class=args.by_class, plot=not args.no_plot)


if __name__ == "__main__":
    main()
