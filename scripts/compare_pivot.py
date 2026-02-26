import os
import yaml
import pandas as pd
from pathlib import Path
import subprocess

# Paths
CWD = Path("d:/김선도/Python/only_quant")
CONFIG_PATH = CWD / "config" / "indicators.yaml"

def run_backtest(start_date="2026-02-01"):
    print(f"🚀 Running backtest from {start_date}...")
    cmd = ["python", "scripts/run_backtest.py", "--start-date", start_date, "--skip-discovery"]
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(CWD))
    return result

def set_pivot(enabled=True):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['pivot_settings']['enabled'] = enabled
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    print(f"✅ Pivot enabled set to: {enabled}")

def main():
    # 1. Disable Pivot
    set_pivot(False)
    run_backtest()
    df_no = pd.read_csv(CWD / "trading_log.csv")
    df_no.to_csv(CWD / "trading_log_no_pivot_agg.csv", index=False)
    pnl_no = df_no['total_rates_cumpnl'].iloc[-1]
    
    # 2. Enable Pivot
    set_pivot(True)
    run_backtest()
    df_pivot = pd.read_csv(CWD / "trading_log.csv")
    df_pivot.to_csv(CWD / "trading_log_pivot_agg.csv", index=False)
    pnl_pivot = df_pivot['total_rates_cumpnl'].iloc[-1]
    
    print("\n" + "="*40)
    print("   AGGREGATE PIVOT PERFORMANCE")
    print("="*40)
    print(f"Baseline (No Pivot): {pnl_no:.2f}")
    print(f"With Aggregate Pivot: {pnl_pivot:.2f}")
    print(f"Improvement: {pnl_pivot - pnl_no:.2f}")
    
    # Check if positions changed
    pos_cols = [c for c in df_no.columns if '_Pos' in c]
    diff_count = (df_no[pos_cols] != df_pivot[pos_cols]).any(axis=1).sum()
    print(f"Days with Position Differences: {diff_count}")
    print("="*40)

if __name__ == "__main__":
    main()
