"""
Two-Phase Realistic Backtest Script

Phase 1: Pre-discover strategies at each month-end business day using main.py's discovery engine.
Phase 2: Simulate monthly trading using pre-cached strategies with Sharpe 6M >= 1.1 filter.

Usage:
    python scripts/run_backtest.py --start-date 2025-09-01
    python scripts/run_backtest.py --start-date 2025-09-01 --skip-discovery  (Phase 2 only)
"""

import sys
import logging
import argparse
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.factory.strategy_factory import StrategyFactory
from src.portfolio.selector import StrategySelector
from scripts.plot_trading_results import plot_pnl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def asset_sort_key(ticker: str) -> tuple:
    """Sort assets into groups: Rates → FX → Index.

    Groups:
      1 = US rates (TU1, TY1)
      2 = Germany rates (DU1, RX1)
      3 = UK rates (G 1)
      4 = Australia rates (YM1, XM1)
      5 = Japan rates (JB1)
      6 = Korea rates (GVSK - yield basis, kept as-is)
      7 = Europe peripheral / other Comdty (OAT1, IK1)
      8 = FX Curncy (EC1, BP1, JY1, AD1, KRW)
      9 = Equity Index (NQ1)
     10 = Other
    """
    if ticker in ('TU1 Comdty', 'TY1 Comdty'): group = 1
    elif ticker in ('DU1 Comdty', 'RX1 Comdty'): group = 2
    elif ticker == 'G 1 Comdty': group = 3
    elif ticker in ('YM1 Comdty', 'XM1 Comdty'): group = 4
    elif ticker == 'JB1 Comdty': group = 5
    elif 'GVSK' in ticker: group = 6
    elif 'Comdty' in ticker: group = 7          # OAT1, IK1 등 기타 선물
    elif 'NQ' in ticker: group = 9
    elif 'Curncy' in ticker: group = 8
    elif 'Index' in ticker or 'Corp' in ticker:  # 잔여 yield 데이터 (혹시 남아있을 경우)
        group = 7
    else: group = 10
    return (group, ticker)


def get_month_end_business_days(prices_index: pd.DatetimeIndex, 
                                 start_date: str, 
                                 end_date: str = None) -> list:
    """
    Get the last business day of each month from price data index.
    Returns dates starting from the month BEFORE start_date (for initial discovery)
    up to end_date.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else prices_index[-1]
    
    # We need discovery starting from the month BEFORE the backtest start
    # e.g., if start_date is 2025-09-01, we need discovery at 2025-08-29 (last biz day of Aug)
    discovery_start = start_dt - pd.DateOffset(months=1)
    
    # Filter to relevant range
    relevant_dates = prices_index[(prices_index >= discovery_start) & (prices_index <= end_dt)]
    
    # Group by year-month and take the last date in each group
    month_ends = relevant_dates.to_series().groupby(
        [relevant_dates.year, relevant_dates.month]
    ).last().values
    
    return sorted([pd.Timestamp(d) for d in month_ends])


def load_config() -> dict:
    """Load backtest thresholds from indicators.yaml."""
    config_path = Path(__file__).parent.parent / 'config' / 'indicators.yaml'
    defaults = {
        'sharpe_threshold_3y': 1.0,
        'sortino_threshold_3y': 1.0,
        'sharpe_threshold_6m': 0.5,
        'sortino_threshold_6m': 0.5,
    }
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            bt_cfg = config.get('backtest', {})
            for key in defaults:
                defaults[key] = bt_cfg.get(key, defaults[key])
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# Signal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_raw_signals(raw_signals: list, prices: pd.DataFrame,
                            prev_date, curr_date) -> dict:
    """
    Re-aggregate a (possibly filtered) list of raw strategy signals into
    per-asset {position, raw_position} using sharpe-weighted voting,
    same logic as selector.aggregate_positions.

    Returns dict: {asset: {'position': int, 'raw_position': float}}
    """
    from collections import defaultdict
    by_asset = defaultdict(list)
    for sig in raw_signals:
        by_asset[sig['asset']].append(sig)

    result = {}
    for asset, sigs in by_asset.items():
        weights = [s['sharpe_6m'] for s in sigs]
        positions = [s['position'] for s in sigs]
        total_w = sum(weights)
        if total_w > 0:
            avg = sum(p * w for p, w in zip(positions, weights)) / total_w
        else:
            avg = sum(positions) / len(positions) if positions else 0.0
        avg = float(np.clip(avg, -1.0, 1.0))
        result[asset] = {'position': avg, 'raw_position': avg}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Pre-Discovery
# ─────────────────────────────────────────────────────────────────────────────
def run_phase1_discovery(prices: pd.DataFrame, month_ends: list, factory_base: Path):
    """
    Pre-discover strategies at each month-end business day.
    Uses main.py's GlobalMacroTradingSystem for reliable discovery.
    """
    from main import GlobalMacroTradingSystem
    
    total = len(month_ends)
    for i, month_end in enumerate(month_ends, 1):
        date_str = month_end.strftime('%Y-%m-%d')
        storage_dir = factory_base / f"strategies_{date_str}"
        
        # Skip if already cached
        if storage_dir.exists() and (storage_dir / 'index.json').exists():
            existing = StrategyFactory(storage_dir=str(storage_dir))
            count = len(existing.index.get('strategies', {}))
            logger.info(f"[{i}/{total}] 📂 Cache exists for {date_str}: {count} strategies. Skipping.")
            continue
        
        logger.info(f"[{i}/{total}] 🔍 Discovering strategies for {date_str}...")
        
        system = GlobalMacroTradingSystem(storage_dir=str(storage_dir))
        result = system.run_discovery(
            start_date="2020-01-01",
            end_date=date_str,
        )
        
        logger.info(f"[{i}/{total}] ✅ {date_str}: Stored {result['stored_strategies']}, Active {result['active_strategies']}")
    
    print(f"\n✅ Phase 1 complete! Strategies pre-discovered for {total} month-end dates.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Simulation
# ─────────────────────────────────────────────────────────────────────────────
def run_phase2_simulation(prices: pd.DataFrame,
                           month_ends: list,
                           factory_base: Path,
                           start_date_str: str,
                           max_correlation: float = 0.3,
                           dd_threshold: float = 10.0,
                           pnl_ma: int = 0):
    """
    Simulate monthly trading using pre-cached strategies.
    Each month uses the previous month-end's strategies.

    dd_threshold: Portfolio drawdown (in PnL units) at which positions scale to 0.
                  Scale = max(0, 1 - drawdown / dd_threshold). 0 = disabled.
    pnl_ma: MA period for per-strategy PnL filter (0 = disabled).
            Strategies whose cumulative PnL is below its own N-day MA are
            silenced (position set to 0) before position aggregation.
    """
    cfg = load_config()
    
    all_dates = prices.index
    target_dates = all_dates[all_dates >= pd.to_datetime(start_date_str)]
    
    if target_dates.empty:
        print(f"❌ No data available from {start_date_str}")
        return
    
    # Sort assets
    all_assets = sorted(list(prices.columns), key=asset_sort_key)
    rates_assets = [a for a in all_assets if asset_sort_key(a)[0] <= 7]
    fx_assets = [a for a in all_assets if asset_sort_key(a)[0] == 8]
    index_assets = [a for a in all_assets if asset_sort_key(a)[0] == 9]
    
    # Build a map: for each month in the backtest, which strategy folder to use?
    # month_ends is sorted; the first one is the "pre-start" discovery date
    strategy_schedule = {}  # month_number -> strategy_folder_date
    for i in range(len(month_ends)):
        strategy_schedule[month_ends[i]] = month_ends[i]
    
    # Track state
    cum_pnl = {asset: 0.0 for asset in all_assets}
    cum_pnl_mom = 0.0   # Momentum strategies
    cum_pnl_mr  = 0.0   # Mean-Reversion strategies
    cum_pnl_adv = 0.0   # Advanced strategies
    log_data = []

    # Drawdown scaling state
    portfolio_cum_pnl = 0.0
    portfolio_hwm = 0.0
    current_selector = None
    current_strategy_date = None

    # Per-strategy PnL MA filter state
    # strat_cum_pnl[sid]     : running cumulative PnL of that strategy
    # strat_cum_history[sid] : deque of last `pnl_ma` cumulative PnL values
    strat_cum_pnl: Dict[str, float] = defaultdict(float)
    strat_cum_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=pnl_ma if pnl_ma > 0 else 1))
    
    print(f"🚀 Running Phase 2 Simulation (Start: {start_date_str}, MaxCorr: {max_correlation})...")
    
    for date in target_dates:
        prev_idx = all_dates.get_loc(date) - 1
        if prev_idx < 0:
            continue
        prev_date = all_dates[prev_idx]
        
        # Determine which strategy folder to use for this date
        # Find the most recent month-end that is <= prev_date
        applicable_month_end = None
        for me in month_ends:
            if me <= prev_date:
                applicable_month_end = me
            else:
                break
        
        if applicable_month_end is None:
            continue
        
        # Rebalance if strategy date changed
        if applicable_month_end != current_strategy_date:
            date_str = applicable_month_end.strftime('%Y-%m-%d')
            storage_dir = factory_base / f"strategies_{date_str}"
            
            if not storage_dir.exists() or not (storage_dir / 'index.json').exists():
                logger.warning(f"⚠️ No strategies found for {date_str}. Run Phase 1 first!")
                continue
            
            factory = StrategyFactory(storage_dir=str(storage_dir))
            # Batch-activate all stored strategies in memory, then save once
            for sid, info in factory.index.get('strategies', {}).items():
                info['is_active'] = True
                # Also update the individual JSON file's is_active field
                strat = factory.load_strategy(sid)
                if strat:
                    strat['is_active'] = True
                    strat_file = factory.storage_dir / f"{sid}.json"
                    with open(strat_file, 'w', encoding='utf-8') as jf:
                        import json as _json
                        _json.dump(strat, jf, indent=2, ensure_ascii=False)
            factory._save_index()  # single write
            
            active_count = sum(1 for v in factory.index['strategies'].values() if v.get('is_active'))
            logger.info(f"📊 Rebalance at {date_str}: {active_count} active strategies")
            
            prev_carry = current_selector._carry_positions.copy() if current_selector else {}
            current_selector = StrategySelector(factory=factory)
            current_selector._carry_positions = prev_carry  # carry-forward across month boundaries

            # Refresh with correlation filter
            prices_until = prices[prices.index <= applicable_month_end]
            current_selector._refresh_sync(prices_until, max_correlation=max_correlation)
            
            current_strategy_date = applicable_month_end
        
        if current_selector is None:
            continue
        
        # Generate signals
        context_prices = prices.loc[:prev_date]
        report = current_selector.get_trading_report(
            context_prices,
            target_date=str(date.date()),
            max_correlation=max_correlation
        )
        raw = report.get('raw_signals', [])

        # ── Per-strategy PnL MA filter ────────────────────────────────────
        pnl_ma_off_count = 0
        if pnl_ma > 0 and raw:
            passing = []
            for sig in raw:
                sid = sig['strategy_id']
                hist = strat_cum_history[sid]
                # Need at least pnl_ma observations to make a judgment
                if len(hist) < pnl_ma:
                    passing.append(sig)   # warm-up: always include
                else:
                    ma_val = sum(hist) / len(hist)
                    if strat_cum_pnl[sid] >= ma_val:
                        passing.append(sig)  # ON: above MA
                    else:
                        pnl_ma_off_count += 1  # OFF: below MA
            # Re-aggregate from passing signals only
            filtered_map = _aggregate_raw_signals(passing, prices, prev_date, date)
            sig_map = {asset: info for asset, info in filtered_map.items()}
        else:
            sig_map = {item['asset']: item for item in report['aggregated_positions']}
        
        # Drawdown scaling: compute scale from prior day's portfolio PnL
        if dd_threshold > 0:
            drawdown = max(0.0, portfolio_hwm - portfolio_cum_pnl)
            dd_scale = max(0.0, 1.0 - drawdown / dd_threshold)
        else:
            dd_scale = 1.0

        # Calculate daily PnL
        row = {'Date': date.date()}
        day_total_pnl = 0.0

        for asset in all_assets:
            p_prev = prices.loc[prev_date, asset]
            p_curr = prices.loc[date, asset]

            sig_info = sig_map.get(asset, {'position': 0})
            pos = sig_info.get('position', 0) * dd_scale

            # 한국 금리(GVSK)만 yield → bps 변화로 계산; 선물(Comdty)/FX/Index는 % 수익률
            if ("Index" in asset or "Corp" in asset) and "NQ" not in asset:
                daily_pnl = (p_curr - p_prev) * 100 * pos  # bps (GVSK yield 기반)
            else:
                daily_pnl = (p_curr / p_prev - 1) * 100 * pos  # % (선물 가격 기반)

            cum_pnl[asset] += daily_pnl
            day_total_pnl += daily_pnl

            row[f"{asset}_Pos"] = round(pos, 4)
            row[f"{asset}_Price"] = round(p_curr, 4)
            row[f"{asset}_CumPnL"] = round(cum_pnl[asset], 2)

        # Update portfolio HWM
        portfolio_cum_pnl += day_total_pnl
        portfolio_hwm = max(portfolio_hwm, portfolio_cum_pnl)
        row['dd_scale'] = round(dd_scale, 4)

        # Summary PnLs
        row['total_rates_cumpnl'] = round(sum(cum_pnl[a] for a in rates_assets), 2)
        row['total_fx_cumpnl'] = round(sum(cum_pnl[a] for a in fx_assets), 2)
        row['total_index_cumpnl'] = round(sum(cum_pnl[a] for a in index_assets), 2)
        
        # ------------------------------------------------------------------
        # Update per-strategy cumulative PnL history (for MA filter on next day)
        # ------------------------------------------------------------------
        for sig in raw:
            sid   = sig['strategy_id']
            asset = sig['asset']
            if asset not in prices.columns:
                continue
            p_p = prices.loc[prev_date, asset]
            p_c = prices.loc[date, asset]
            pos_s   = sig['position']
            is_rate = ("Index" in asset or "Corp" in asset) and "NQ" not in asset
            dpnl_s  = (p_c - p_p) * 100 * pos_s if is_rate else (p_c / p_p - 1) * 100 * pos_s
            strat_cum_pnl[sid] += dpnl_s
            strat_cum_history[sid].append(strat_cum_pnl[sid])

        row['pnl_ma_off_count'] = pnl_ma_off_count

        # ------------------------------------------------------------------
        # Per-strategy-type daily PnL  (MOM / MR / ADV)
        # ------------------------------------------------------------------
        daily_mom = daily_mr = daily_adv = 0.0
        n_mom = n_mr = n_adv = 0
        for sig in raw:
            asset = sig['asset']
            if asset not in prices.columns:
                continue
            p_prev_s = prices.loc[prev_date, asset]
            p_curr_s = prices.loc[date, asset]
            pos_s    = sig['position']
            is_rate  = ("Index" in asset or "Corp" in asset) and "NQ" not in asset
            if is_rate:
                dpnl = (p_curr_s - p_prev_s) * 100 * pos_s
            else:
                dpnl = (p_curr_s / p_prev_s - 1) * 100 * pos_s
            stype = sig.get('strategy_type', '')
            if stype == 'momentum':
                daily_mom += dpnl; n_mom += 1
            elif stype == 'mean_reversion':
                daily_mr  += dpnl; n_mr  += 1
            elif stype == 'advanced':
                daily_adv += dpnl; n_adv += 1
        # Normalise by number of strategies (so scale is comparable to per-asset PnL)
        cum_pnl_mom += (daily_mom / n_mom if n_mom else 0.0)
        cum_pnl_mr  += (daily_mr  / n_mr  if n_mr  else 0.0)
        cum_pnl_adv += (daily_adv / n_adv if n_adv else 0.0)

        row['total_mom_cumpnl'] = round(cum_pnl_mom, 4)
        row['total_mr_cumpnl']  = round(cum_pnl_mr,  4)
        row['total_adv_cumpnl'] = round(cum_pnl_adv, 4)

        log_data.append(row)
    
    # ─── Save Results ────────────────────────────────────────────────────────
    if not log_data:
        print("❌ No trading data generated.")
        return
    
    df_log = pd.DataFrame(log_data)
    df_log.set_index('Date', inplace=True)
    
    # Reorder columns
    total_pnl_cols = ['total_rates_cumpnl', 'total_fx_cumpnl', 'total_index_cumpnl',
                      'total_mom_cumpnl', 'total_mr_cumpnl', 'total_adv_cumpnl']
    cum_pnl_cols = [c for c in df_log.columns if '_CumPnL' in c]
    other_cols = [c for c in df_log.columns if c not in total_pnl_cols + cum_pnl_cols]
    df_log = df_log[total_pnl_cols + cum_pnl_cols + other_cols]
    
    output_path = "trading_log.csv"
    df_log.to_csv(output_path)
    print(f"✅ Phase 2 complete! Trading log saved to {output_path} ({len(df_log)} rows)")
    
    # Generate PnL plot
    dd_tag = f"_dd{int(dd_threshold)}" if dd_threshold > 0 else ""
    ma_tag = f"_ma{pnl_ma}" if pnl_ma > 0 else ""
    try:
        plot_pnl(max_correlation=max_correlation, mode=f'backtest{dd_tag}{ma_tag}',
                 start_date=start_date_str,
                 end_date=prices.index[-1].strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Two-Phase Realistic Backtest')
    parser.add_argument('--start-date', default='2025-09-01', 
                        help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=None,
                        help='Backtest end date (YYYY-MM-DD, default: latest)')
    parser.add_argument('--max-corr', type=float, default=0.5,
                        help='Maximum correlation threshold')
    parser.add_argument('--skip-discovery', action='store_true',
                        help='Skip Phase 1 (use existing cached strategies)')
    parser.add_argument('--dd-threshold', type=float, default=0.0,
                        help='Drawdown scaling threshold in PnL units (0 = disabled). '
                             'Positions scale linearly to 0 as portfolio drawdown reaches this value.')
    parser.add_argument('--pnl-ma', type=int, default=0,
                        help='Per-strategy PnL MA filter period (0 = disabled). '
                             'Strategies below their own N-day cumulative-PnL MA are silenced.')
    
    args = parser.parse_args()
    
    # Load data
    print("📊 Loading price data...")
    loader = DataLoader()
    prices_raw = loader.load_data(use_cache=True)
    preprocessor = DataPreprocessor(prices_raw)
    prices = preprocessor.clean().get_data()
    
    factory_base = Path(__file__).parent.parent / 'src' / 'factory'
    factory_base.mkdir(exist_ok=True)
    
    # Get month-end business days
    month_ends = get_month_end_business_days(prices.index, args.start_date, args.end_date)
    
    print(f"\n📅 Month-end dates for discovery:")
    for d in month_ends:
        print(f"   {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})")
    
    # Phase 1: Discovery
    if not args.skip_discovery:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: Strategy Pre-Discovery")
        print(f"{'='*60}")
        run_phase1_discovery(prices, month_ends, factory_base)
    else:
        print("\n⏭️  Skipping Phase 1 (--skip-discovery)")
    
    # Phase 2: Simulation
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Trading Simulation")
    print(f"{'='*60}")
    if args.dd_threshold > 0:
        print(f"📉 Drawdown scaling enabled: threshold = {args.dd_threshold}")
    if args.pnl_ma > 0:
        print(f"🔍 PnL MA filter enabled: period = {args.pnl_ma} days")
    run_phase2_simulation(
        prices, month_ends, factory_base,
        start_date_str=args.start_date,
        max_correlation=args.max_corr,
        dd_threshold=args.dd_threshold,
        pnl_ma=args.pnl_ma,
    )


if __name__ == "__main__":
    main()
