import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_pnl():
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

    fig, (ax1, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(14, 24), sharex=True)

    # --- Plot 1: Aggregate Rates vs FX (Dual Y) ---
    color_rates = 'tab:blue'
    ax1.set_ylabel('Total Rates Cum PnL (bps)', color=color_rates, fontweight='bold')
    ax1.plot(df['Date'], df['total_rates_cumpnl'], color=color_rates, linewidth=3, label='Total Rates (bps)')
    ax1.tick_params(axis='y', labelcolor=color_rates)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color_fx = 'tab:orange'
    ax2.set_ylabel('Total FX Cum PnL (%)', color=color_fx, fontweight='bold')
    ax2.plot(df['Date'], df['total_fx_cumpnl'], color=color_fx, linewidth=3, label='Total FX (%)')
    ax2.tick_params(axis='y', labelcolor=color_fx)
    
    # Add Total Index to ax2 or a separate axis? Let's add to ax2 since it's also %
    ax2.plot(df['Date'], df['total_index_cumpnl'], color='tab:green', linewidth=3, label='Total Index (%)', linestyle='--')
    
    ax1.set_title('Global Macro Strategy: Aggregate Performance', fontsize=16, pad=20)

    # --- Plot 2: Individual Rates (BPS) ---
    for col in rate_cols:
        label = col.replace('_CumPnL', '')
        ax3.plot(df['Date'], df[col], label=label, alpha=0.8)
    ax3.set_ylabel('Individual Rates (bps)', fontweight='bold')
    ax3.set_title('Individual Rates Performance', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    # --- Plot 3: Individual FX (%) ---
    for col in fx_cols:
        label = col.replace('_CumPnL', '')
        ax4.plot(df['Date'], df[col], label=label, alpha=0.8)
    ax4.set_ylabel('Individual FX (%)', fontweight='bold')
    ax4.set_title('Individual FX Performance', fontsize=14)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    # --- Plot 4: Individual Indices (%) ---
    for col in index_cols:
        label = col.replace('_CumPnL', '')
        ax5.plot(df['Date'], df[col], label=label, alpha=0.8)
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

    output_path = "trading_pnl.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Success! PnL plot saved to {output_path}")

if __name__ == "__main__":
    plot_pnl()
