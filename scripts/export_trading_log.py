import os
import logging
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.portfolio.selector import StrategySelector
from src.data.preprocessor import DataPreprocessor
from src.factory.strategy_factory import StrategyFactory
from src.backtester.batch_explorer import BatchExplorer
from scripts.plot_trading_results import plot_pnl

import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_sortino(returns: pd.Series) -> float:
    """Calculate Sortino ratio safely."""
    if len(returns) < 2:
        return 0.0
    mean = returns.mean() * 252
    downside_returns = returns[returns < 0]
    if len(downside_returns) < 2:
        return 0.0
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(mean / downside_std)

def get_strategies_for_date(date, prices, storage_dir_base, sharpe_3y_min=1.0, sharpe_6m_min=1.1, sortino_3y_min=1.0):
    """
    Get or discover strategies for a specific date (T).
    Uses data up to T to find qualified strategies.
    Stores them in factory/strategies_{T}.
    """
    date_str = date.strftime('%Y-%m-%d')
    storage_dir = Path(storage_dir_base) / f"strategies_{date_str}"
    
    factory = StrategyFactory(storage_dir=str(storage_dir))
    
    # Check if we already have indexed strategies in this folder
    if storage_dir.exists() and (storage_dir / 'index.json').exists():
        logger.info(f"📂 Loading {len(factory.get_all_strategies())} cached strategies from {storage_dir.name}")
        factory.activate_qualified_strategies(sharpe_6m_min)
        return factory
        
    logger.info(f"🔍 No cache found for {date_str}. Running Discovery (Realistic)...")
    
    # Discovery using data up to this date
    prices_until = prices[prices.index <= date]
    explorer = BatchExplorer(prices_until)
    
    # 1. Discover ALL possible returns
    all_rets_df, strat_configs = explorer.evaluate_all_strategies()
    
    # 2. Calculate Sharpe and Sortino Ratios
    # 3Y (or all data)
    means = all_rets_df.mean() * 252
    stds = all_rets_df.std() * np.sqrt(252)
    sharpes_3y = means / stds.replace(0, np.nan)
    
    # Sortino 3Y
    sortinos_3y = all_rets_df.apply(safe_sortino)
    
    # 6M slice (approx 126 days)
    slice_6m = all_rets_df.iloc[-126:] if len(all_rets_df) >= 126 else all_rets_df
    means_6m = slice_6m.mean() * 252
    stds_6m = slice_6m.std() * np.sqrt(252)
    sharpes_6m = means_6m / stds_6m.replace(0, np.nan)
    
    # 3. Filter and Save
    stored_count = 0
    for sid, config in strat_configs.items():
        s3y = sharpes_3y.get(sid, 0)
        sort3y = sortinos_3y.get(sid, 0)
        s6m = sharpes_6m.get(sid, 0)
        
        # Check double criteria: Sharpe 3Y + Sortino 3Y
        if s3y >= sharpe_3y_min and sort3y >= sortino_3y_min:
            performance = {
                'sharpe_3y': float(s3y),
                'sortino_3y': float(sort3y),
                'sharpe_6m': float(s6m),
                'num_trades': 20, # BatchExplorer/pf.stats could be more precise but using 20 as placeholder for qualifier
                # Add other metrics if needed
            }
            factory.save_strategy(config, performance)
            stored_count += 1
            
            # Activate if 6M threshold met
            if s6m >= sharpe_6m_min:
                factory.set_active(sid, True)
                
    logger.info(f"✅ Discovery complete for {date_str}: {stored_count} strategies saved to {storage_dir.name}")
    return factory

def export_trading_log(start_date_str="2026-01-01", max_correlation=0.3, mode='case_b'):
    loader = DataLoader()
    prices_raw = loader.load_data(use_cache=True)
    
    # Preprocess data
    preprocessor = DataPreprocessor(prices_raw)
    prices = preprocessor.clean().get_data()
    
    # Load thresholds from config
    config_path = Path(__file__).parent.parent / 'config' / 'indicators.yaml'
    s3y_min = 1.0
    s6m_min = 1.1
    sort3y_min = 1.0
    if config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            bt_cfg = config.get('backtest', {})
            s3y_min = bt_cfg.get('sharpe_threshold_3y', s3y_min)
            s6m_min = bt_cfg.get('sharpe_threshold_6m', s6m_min)
            sort3y_min = bt_cfg.get('sortino_threshold_3y', sort3y_min)

    factory_base = Path(__file__).parent.parent / 'factory'
    factory_base.mkdir(exist_ok=True)
    
    # 1. Dates
    all_dates = prices.index
    target_dates = all_dates[all_dates >= pd.to_datetime(start_date_str)]
    
    if not target_dates.empty:
        print(f"No data available from {start_date_str}")
        return

    # custom sorting logic
    def asset_sort_key(ticker: str) -> tuple:
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

    all_assets = sorted(list(prices.columns), key=asset_sort_key)
    
    # Track cumulative PnL
    cum_pnl = {asset: 0.0 for asset in all_assets}
    
    # Identify Rates vs FX vs Index assets for aggregation
    rates_assets = [a for a in all_assets if asset_sort_key(a)[0] <= 7 and asset_sort_key(a)[0] != 9] # All except NQ Index and FX
    fx_assets = [a for a in all_assets if asset_sort_key(a)[0] == 8]
    index_assets = [a for a in all_assets if asset_sort_key(a)[0] == 9] # NQ Index
    
    log_data = []

    print(f"🚀 Running Realistic Backtest (Mode: {mode}, S3Y: {s3y_min}, S6M: {s6m_min}, Sort3Y: {sort3y_min})...")
    
    # Current active selector (will be updated on rebalance)
    current_selector = None
    last_rebalance_month = -1
    
    for date in target_dates:
        prev_idx = all_dates.get_loc(date) - 1
        if prev_idx < 0:
            # First day in list - we can't calculate PnL yet but can set initial positions if we want.
            # However, for a clean T/T-1 log, we start from the day we have a previous price.
            continue
            
        prev_date = all_dates[prev_idx]
        
        # REBALANCE CHECK
        should_rebalance = False
        if current_selector is None:
            should_rebalance = True # Initialize
        elif mode == 'case_b' and prev_date.month != last_rebalance_month:
            should_rebalance = True # Monthly rebalance
            
        if should_rebalance:
            # Rebalance at the PREVIOUS day's close (T-1) to trade at T's close
            rebalance_point = prev_date
            factory = get_strategies_for_date(rebalance_point, prices, factory_base, s3y_min, s6m_min, sort3y_min)
            current_selector = StrategySelector(factory=factory, sharpe_6m_threshold=s6m_min)
            last_rebalance_month = rebalance_point.month
            
            # Important: Pre-refresh strategies for the selector with correlation filter
            # Using data up to rebalance_point
            prices_until = prices[prices.index <= rebalance_point]
            current_selector._refresh_sync(prices_until, max_correlation=max_correlation)
        
        # Determine signals for 'date' using data up to 'prev_date'
        context_prices = prices.loc[:prev_date]
        report = current_selector.get_trading_report(
            context_prices, 
            target_date=str(date.date()),
            max_correlation=max_correlation
        )
        sig_map = {item['asset']: item for item in report['aggregated_positions']}
        
        row = {'Date': date.date()}
        
        for asset in all_assets:
            p_prev = prices.loc[prev_date, asset]
            p_curr = prices.loc[date, asset]
            
            sig_info = sig_map.get(asset, {'position': 0})
            pos = sig_info.get('position', 0)
            
            # Calculate daily PnL: 
            # 1. Rates (Index/Corp except NQ) use (Prev - Curr) bps
            if ("Index" in asset or "Corp" in asset) and "NQ" not in asset:
                daily_pnl = (p_curr - p_prev) * 100 * pos  # bps
            # 2. FX and Equity Indices use (Curr / Prev - 1) %
            else:
                daily_pnl = (p_curr / p_prev - 1) * 100 * pos  # %
                
            cum_pnl[asset] += daily_pnl
            
            # Add to row with logical ordering
            row[f"{asset}_Pos"] = pos
            row[f"{asset}_Price"] = round(p_curr, 4)
            row[f"{asset}_CumPnL"] = round(cum_pnl[asset], 2)
            
        # Calculate summary PnLs
        row['total_rates_cumpnl'] = round(sum(cum_pnl[a] for a in rates_assets), 2)
        row['total_fx_cumpnl'] = round(sum(cum_pnl[a] for a in fx_assets), 2)
        row['total_index_cumpnl'] = round(sum(cum_pnl[a] for a in index_assets), 2)
            
        log_data.append(row)

    # Convert to DataFrame
    df_log = pd.DataFrame(log_data)
    df_log.set_index('Date', inplace=True)
    
    # Reorder columns: Total PnLs, then Asset CumPnLs, then Pos and Price
    total_pnl_cols = ['total_rates_cumpnl', 'total_fx_cumpnl', 'total_index_cumpnl']
    cum_pnl_cols = [c for c in df_log.columns if '_CumPnL' in c]
    other_cols = [c for c in df_log.columns if c not in total_pnl_cols + cum_pnl_cols]
    
    df_log = df_log[total_pnl_cols + cum_pnl_cols + other_cols]
    
    # Export
    output_path = "trading_log.csv"
    df_log.to_csv(output_path)
    print(f"✅ Success! Realistic Trading log saved to {output_path} ({len(df_log)} rows)")
    
    # Automatically update the PnL plot
    try:
        plot_pnl(max_correlation=max_correlation, mode=mode)
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Realistic Trading Log')
    parser.add_argument('--start-date', default='2026-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--mode', choices=['case_a', 'case_b'], default='case_b', help='Backtest mode: static(a) or monthly(b)')
    parser.add_argument('--max-corr', type=float, default=0.3, help='Maximum correlation threshold')
    
    args = parser.parse_args()
    
    export_trading_log(
        start_date_str=args.start_date, 
        max_correlation=args.max_corr, 
        mode=args.mode
    )
