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

def export_trading_log(start_date_str="2026-02-01"):
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
        elif 'Index' in ticker or 'Corp' in ticker: group = 7
        elif 'Curncy' in ticker: group = 8
        else: group = 9
        return (group, ticker)

    all_assets = sorted(list(prices.columns), key=asset_sort_key)
    
    # Track cumulative PnL
    cum_pnl = {asset: 0.0 for asset in all_assets}
    log_data = []

    print(f"🔄 Generating log since {start_date_str}...")
    
    for date in target_dates:
        prev_idx = all_dates.get_loc(date) - 1
        if prev_idx < 0:
            # First day in list - we can't calculate PnL yet but can set initial positions if we want.
            # However, for a clean T/T-1 log, we start from the day we have a previous price.
            continue
            
        prev_date = all_dates[prev_idx]
        context_prices = prices.loc[:prev_date]
        
        # Determine signals for 'date' using data up to 'prev_date'
        report = selector.get_trading_report(context_prices, target_date=str(date.date()))
        sig_map = {item['asset']: item for item in report['aggregated_positions']}
        
        row = {'Date': date.date()}
        
        for asset in all_assets:
            p_prev = prices.loc[prev_date, asset]
            p_curr = prices.loc[date, asset]
            
            sig_info = sig_map.get(asset, {'position': 0})
            pos = sig_info.get('position', 0)
            
            # Calculate daily PnL: Rates use (Prev - Curr) so Yield Up = Loss
            if "Index" in asset or "Corp" in asset:
                daily_pnl = (p_prev - p_curr) * 100 * pos  # bps
            else:
                daily_pnl = (p_curr / p_prev - 1) * 100 * pos  # %
                
            cum_pnl[asset] += daily_pnl
            
            # Add to row with logical ordering
            row[f"{asset}_Pos"] = pos
            row[f"{asset}_Price"] = round(p_curr, 4)
            row[f"{asset}_CumPnL"] = round(cum_pnl[asset], 2)
            
        log_data.append(row)

    # Convert to DataFrame
    df_log = pd.DataFrame(log_data)
    df_log.set_index('Date', inplace=True)
    
    # Reorder columns: CumPnL first, then Pos and Price
    cum_pnl_cols = [c for c in df_log.columns if '_CumPnL' in c]
    other_cols = [c for c in df_log.columns if '_CumPnL' not in c]
    df_log = df_log[cum_pnl_cols + other_cols]
    
    # Export
    output_path = "trading_log.csv"
    df_log.to_csv(output_path)
    print(f"✅ Success! Trading log saved to {output_path} ({len(df_log)} rows)")

if __name__ == "__main__":
    export_trading_log()
