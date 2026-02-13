import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_pnl(max_correlation=None, mode=None):
    csv_path = Path("trading_log.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run export_trading_log.py first.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)

    # Asset columns
    rate_cols = [c for c in df.columns if '_CumPnL' in c and 'Curncy' not in c and 'NQ' not in c]
    fx_cols = [c for c in df.columns if '_CumPnL' in c and 'Curncy' in c]
    index_cols = [c for c in df.columns if '_CumPnL' in c and 'NQ' in c]

    # Calculate Sharpe Ratios
    def calc_sharpe(series):
        daily_rets = series.diff().dropna()
        if daily_rets.std() == 0 or len(daily_rets) < 2:
            return 0.0
        return (daily_rets.mean() / daily_rets.std()) * (252**0.5)

    sharpe_rates = calc_sharpe(df['total_rates_cumpnl'])
    sharpe_fx = calc_sharpe(df['total_fx_cumpnl'])
    sharpe_idx = calc_sharpe(df['total_index_cumpnl'])

    fig, (ax1, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(14, 24), sharex=True)

    # --- Plot 1: Aggregate Rates vs FX (Dual Y) ---
    color_rates = 'tab:blue'
    ax1.set_ylabel('Total Rates Cum PnL (bps)', color=color_rates, fontweight='bold')
    ax1.plot(df['Date'], df['total_rates_cumpnl'], color=color_rates, linewidth=3, label=f'Total Rates (Sharpe: {sharpe_rates:.2f})')
    ax1.tick_params(axis='y', labelcolor=color_rates)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate last value
    last_date = df['Date'].iloc[-1]
    last_val_rates = df['total_rates_cumpnl'].iloc[-1]
    ax1.text(last_date, last_val_rates, f" {last_val_rates:.1f}", color=color_rates, fontweight='bold', va='center')

    ax2 = ax1.twinx()
    color_fx = 'tab:orange'
    ax2.set_ylabel('Total FX Cum PnL (%)', color=color_fx, fontweight='bold')
    ax2.plot(df['Date'], df['total_fx_cumpnl'], color=color_fx, linewidth=3, label=f'Total FX (Sharpe: {sharpe_fx:.2f})')
    ax2.tick_params(axis='y', labelcolor=color_fx)
    
    # Annotate last values for ax2
    last_val_fx = df['total_fx_cumpnl'].iloc[-1]
    ax2.text(last_date, last_val_fx, f" {last_val_fx:.2f}%", color=color_fx, fontweight='bold', va='center')
    
    last_val_idx = df['total_index_cumpnl'].iloc[-1]
    ax2.text(last_date, last_val_idx, f" {last_val_idx:.2f}%", color='tab:green', fontweight='bold', va='center')
    
    # Add Total Index to ax2 or a separate axis? Let's add to ax2 since it's also %
    ax2.plot(df['Date'], df['total_index_cumpnl'], color='tab:green', linewidth=3, label=f'Total Index (Sharpe: {sharpe_idx:.2f})', linestyle='--')
    
    title_mode = f" (Mode: {mode.upper()})" if mode else ""
    ax1.set_title(f'Global Macro Strategy: Aggregate Performance{title_mode}', fontsize=16, pad=20)

    # --- Plot 2: Individual Rates (BPS) ---
    for col in rate_cols:
        label = col.replace('_CumPnL', '')
        line = ax3.plot(df['Date'], df[col], label=label, alpha=0.8)[0]
        ax3.text(last_date, df[col].iloc[-1], f" {df[col].iloc[-1]:.1f}", color=line.get_color(), fontsize=8, va='center')
    ax3.set_ylabel('Individual Rates (bps)', fontweight='bold')
    ax3.set_title('Individual Rates Performance', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    # --- Plot 3: Individual FX (%) ---
    for col in fx_cols:
        label = col.replace('_CumPnL', '')
        line = ax4.plot(df['Date'], df[col], label=label, alpha=0.8)[0]
        ax4.text(last_date, df[col].iloc[-1], f" {df[col].iloc[-1]:.2f}%", color=line.get_color(), fontsize=8, va='center')
    ax4.set_ylabel('Individual FX (%)', fontweight='bold')
    ax4.set_title('Individual FX Performance', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    # --- Plot 4: Individual Indices (%) ---
    for col in index_cols:
        label = col.replace('_CumPnL', '')
        line = ax5.plot(df['Date'], df[col], label=label, alpha=0.8)[0]
        ax5.text(last_date, df[col].iloc[-1], f" {df[col].iloc[-1]:.2f}%", color=line.get_color(), fontsize=8, va='center')
    ax5.set_ylabel('Individual Indices (%)', fontweight='bold')
    ax5.set_title('Individual Index Performance', fontsize=14)
    ax5.grid(True, linestyle='--', alpha=0.6)
    ax5.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    # Global Formatting
    plt.xlabel('Date')
    fig.tight_layout()
    
    # Legend for top plot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    suffix_corr = f"_corr{max_correlation}" if max_correlation is not None else ""
    suffix_mode = f"_{mode}" if mode else ""
    suffix_sharpe = f"_rates{sharpe_rates:.2f}_fx{sharpe_fx:.2f}"
    output_path = f"trading_pnl{suffix_corr}{suffix_mode}{suffix_sharpe}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Success! PnL plot saved to {output_path}")

if __name__ == "__main__":
    plot_pnl()
