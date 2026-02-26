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
    """Sort assets into groups: Rates → FX → Index."""
    if 'USGG' in ticker: group = 1
    elif 'GDBR' in ticker: group = 2
    elif 'GUKG' in ticker: group = 3
    elif 'GTAUD' in ticker: group = 4
    elif 'GJGB' in ticker: group = 5
    elif 'GVSK' in ticker: group = 6
    elif 'Index' in ticker or 'Corp' in ticker:
        if 'NQ' in ticker: group = 9
        else: group = 7
    elif 'Curncy' in ticker: group = 8
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
                           pivot_enabled: Optional[bool] = None):
    """
    Simulate monthly trading using pre-cached strategies.
    Each month uses the previous month-end's strategies.
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
    log_data = []
    current_selector = None
    current_strategy_date = None
    
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
            # Activate all stored strategies (no 6M filter)
            for sid in factory.index.get('strategies', {}):
                factory.set_active(sid, True)
            
            active_count = len(factory.get_active_strategies())
            logger.info(f"📊 Rebalance at {date_str}: {active_count} active strategies")
            
            current_selector = StrategySelector(factory=factory)
            
            # Override pivot setting if provided via CLI
            if pivot_enabled is not None:
                if 'pivot_settings' not in current_selector.config:
                    current_selector.config['pivot_settings'] = {}
                current_selector.config['pivot_settings']['enabled'] = pivot_enabled
                logger.info(f"⚙️ Pivot enabled overridden by CLI: {pivot_enabled}")

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
        sig_map = {item['asset']: item for item in report['aggregated_positions']}
        
        # Calculate daily PnL
        row = {'Date': date.date()}
        
        for asset in all_assets:
            p_prev = prices.loc[prev_date, asset]
            p_curr = prices.loc[date, asset]
            
            sig_info = sig_map.get(asset, {'position': 0})
            pos = sig_info.get('position', 0)
            
            # Rates: bps change
            if ("Index" in asset or "Corp" in asset) and "NQ" not in asset:
                daily_pnl = (p_curr - p_prev) * 100 * pos  # bps
            # FX & Equity Index: % change
            else:
                daily_pnl = (p_curr / p_prev - 1) * 100 * pos  # %
            
            cum_pnl[asset] += daily_pnl
            
            row[f"{asset}_Pos"] = pos
            row[f"{asset}_Price"] = round(p_curr, 4)
            row[f"{asset}_CumPnL"] = round(cum_pnl[asset], 2)
            
        # Log portfolio-level pivot metadata (same for all assets, so we only need one set of columns)
        # Use info from the first asset row which contains the broadcasted portfolio metrics
        first_asset = all_assets[0]
        first_info = sig_map.get(first_asset, {})
        row["Portfolio_Rolling_Sharpe"] = round(first_info.get('portfolio_sharpe', 0), 2)
        row["Portfolio_Pivot_Active"] = 1 if first_info.get('portfolio_pivot_active', False) else 0
        
        # Summary PnLs
        row['total_rates_cumpnl'] = round(sum(cum_pnl[a] for a in rates_assets), 2)
        row['total_fx_cumpnl'] = round(sum(cum_pnl[a] for a in fx_assets), 2)
        row['total_index_cumpnl'] = round(sum(cum_pnl[a] for a in index_assets), 2)
        
        log_data.append(row)
    
    # ─── Save Results ────────────────────────────────────────────────────────
    if not log_data:
        print("❌ No trading data generated.")
        return
    
    df_log = pd.DataFrame(log_data)
    df_log.set_index('Date', inplace=True)
    
    # Reorder columns
    total_pnl_cols = ['total_rates_cumpnl', 'total_fx_cumpnl', 'total_index_cumpnl']
    cum_pnl_cols = [c for c in df_log.columns if '_CumPnL' in c]
    other_cols = [c for c in df_log.columns if c not in total_pnl_cols + cum_pnl_cols]
    df_log = df_log[total_pnl_cols + cum_pnl_cols + other_cols]
    
    output_path = "trading_log.csv"
    df_log.to_csv(output_path)
    print(f"✅ Phase 2 complete! Trading log saved to {output_path} ({len(df_log)} rows)")
    
    # Generate PnL plot
    pivot_status = "pivot_on" if (pivot_enabled if pivot_enabled is not None else True) else "pivot_off"
    try:
        plot_pnl(max_correlation=max_correlation, mode=f'backtest_{pivot_status}', start_date=start_date_str,
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
    parser.add_argument('--pivot', dest='pivot', action='store_true', help='Force enable pivot logic')
    parser.add_argument('--no-pivot', dest='pivot', action='store_false', help='Force disable pivot logic')
    parser.set_defaults(pivot=None)
    
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
    run_phase2_simulation(
        prices, month_ends, factory_base,
        start_date_str=args.start_date,
        max_correlation=args.max_corr,
        pivot_enabled=args.pivot
    )


if __name__ == "__main__":
    main()
