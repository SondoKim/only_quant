import os
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
from scripts.plot_trading_results import plot_pnl

import argparse

def export_trading_log(start_date_str="2026-01-01", max_correlation=0.7):
    loader = DataLoader()
    prices_raw = loader.load_data(use_cache=True)
    
    # Preprocess data
    preprocessor = DataPreprocessor(prices_raw)
    prices = preprocessor.clean().get_data()
    
    # Load config for thresholds
    config_path = Path(__file__).parent.parent / 'config' / 'indicators.yaml'
    sharpe_6m_threshold = 0.7  # default
    if config_path.exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            sharpe_6m_threshold = config.get('backtest', {}).get('sharpe_threshold_6m', 0.7)

    selector = StrategySelector(sharpe_6m_threshold=sharpe_6m_threshold)
    
    start_date = pd.to_datetime(start_date_str)
    all_dates = prices.index
    target_dates = [d for d in all_dates if d >= start_date]
    
    if not target_dates:
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
            if 'NQ' in ticker: group = 9 # Nasdaq is an Index but should be treated as Equity
            else: group = 7
        elif 'Curncy' in ticker: group = 8
        else: group = 10
        return (group, ticker)

    all_assets = sorted(list(prices.columns), key=asset_sort_key)
    
    # Track cumulative PnL
    cum_pnl = {asset: 0.0 for asset in all_assets}
    
    # Identify Rates vs FX vs Index assets for aggregation
    rates_assets = [a for a in all_assets if asset_sort_key(a)[0] <= 7]
    fx_assets = [a for a in all_assets if asset_sort_key(a)[0] == 8]
    index_assets = [a for a in all_assets if asset_sort_key(a)[0] == 9]
    
    log_data = []

    print(f"🔄 Generating log since {start_date_str} (Max Corr: {max_correlation})...")
    
    for date in target_dates:
        prev_idx = all_dates.get_loc(date) - 1
        if prev_idx < 0:
            # First day in list - we can't calculate PnL yet but can set initial positions if we want.
            # However, for a clean T/T-1 log, we start from the day we have a previous price.
            continue
            
        prev_date = all_dates[prev_idx]
        context_prices = prices.loc[:prev_date]
        
        # Determine signals for 'date' using data up to 'prev_date'
        report = selector.get_trading_report(
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
                daily_pnl = (p_prev - p_curr) * 100 * pos  # bps
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
    print(f"✅ Success! Trading log saved to {output_path} ({len(df_log)} rows)")
    
    # Automatically update the PnL plot
    try:
        plot_pnl(max_correlation=max_correlation)
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Trading Log')
    parser.add_argument('--start-date', default='2026-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--max-corr', type=float, default=None, help='Maximum correlation threshold')
    parser.add_argument('--batch', action='store_true', help='Run batch correlation loop (0.1 to 0.7)')
    args = parser.parse_args()
    
    if args.batch:
        # Loop from 0.1 to 0.7 in 0.1 steps
        for corr in np.arange(0.1, 0.8, 0.1):
            corr_val = round(float(corr), 1)
            print(f"\n🚀 Running batch for correlation: {corr_val}")
            export_trading_log(start_date_str=args.start_date, max_correlation=corr_val)
    else:
        # Use provided max-corr or default to 0.7
        corr_val = args.max_corr if args.max_corr is not None else 0.7
        export_trading_log(start_date_str=args.start_date, max_correlation=corr_val)
