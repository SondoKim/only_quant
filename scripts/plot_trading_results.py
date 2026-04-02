import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_pnl(max_correlation=None, mode=None, start_date=None, end_date=None):
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

    last_date = df['Date'].iloc[-1]
    title_mode = f" (Mode: {mode.upper()})" if mode else ""

    # Check if split strategy-type columns exist
    has_split = all(c in df.columns for c in [
        'mom_rates_cumpnl', 'mr_rates_cumpnl', 'adv_rates_cumpnl',
        'mom_fx_cumpnl', 'mr_fx_cumpnl', 'adv_fx_cumpnl',
    ])
    has_alpha = all(c in df.columns for c in [
        'a1_rates_cumpnl', 'a2_rates_cumpnl', 'a3_rates_cumpnl',
        'a1_fx_cumpnl', 'a2_fx_cumpnl', 'a3_fx_cumpnl',
    ])

    # =====================================================================
    # Layout: 1 top (full width) + 2x2 grid (MOM/MR/ADV) + 2x1 (Alpha)
    # 4 rows if alpha data exists, 3 rows otherwise
    # =====================================================================
    n_rows = 4 if has_alpha else 3
    fig = plt.figure(figsize=(18, 6 * n_rows + 4))
    gs = fig.add_gridspec(n_rows, 2, height_ratios=[1] * n_rows, hspace=0.3, wspace=0.25)

    ax_top = fig.add_subplot(gs[0, :])       # Total Return (full width)
    ax_ir  = fig.add_subplot(gs[1, 0])       # Individual Rates
    ax_ifx = fig.add_subplot(gs[1, 1])       # Individual FX
    ax_sr  = fig.add_subplot(gs[2, 0])       # Strategy-Type Rates
    ax_sfx = fig.add_subplot(gs[2, 1])       # Strategy-Type FX
    if has_alpha:
        ax_ar  = fig.add_subplot(gs[3, 0])   # Alpha Rates
        ax_afx = fig.add_subplot(gs[3, 1])   # Alpha FX

    # -----------------------------------------------------------------
    # Top: Total Return (Rates + FX dual Y-axis)
    # -----------------------------------------------------------------
    color_rates = 'tab:blue'
    ax_top.set_ylabel('Total Rates Cum PnL (bps)', color=color_rates, fontweight='bold')
    ax_top.plot(df['Date'], df['total_rates_cumpnl'], color=color_rates, linewidth=2.5,
                label=f'Total Rates (Sharpe: {sharpe_rates:.2f})')
    ax_top.tick_params(axis='y', labelcolor=color_rates)
    ax_top.grid(True, linestyle='--', alpha=0.6)
    last_val_rates = df['total_rates_cumpnl'].iloc[-1]
    ax_top.text(last_date, last_val_rates, f" {last_val_rates:.1f}",
                color=color_rates, fontweight='bold', va='center', fontsize=9)

    ax_top2 = ax_top.twinx()
    color_fx = 'tab:orange'
    ax_top2.set_ylabel('Total FX Cum PnL (%)', color=color_fx, fontweight='bold')
    ax_top2.plot(df['Date'], df['total_fx_cumpnl'], color=color_fx, linewidth=2.5,
                 label=f'Total FX (Sharpe: {sharpe_fx:.2f})')
    ax_top2.tick_params(axis='y', labelcolor=color_fx)
    last_val_fx = df['total_fx_cumpnl'].iloc[-1]
    ax_top2.text(last_date, last_val_fx, f" {last_val_fx:.2f}%",
                 color=color_fx, fontweight='bold', va='center', fontsize=9)

    ax_top.set_title(f'Global Macro Strategy: Total Return{title_mode}', fontsize=16, pad=15)

    # Regime shading: T=trending (green), R=ranging (orange)
    has_regime = ('regime_rates' in df.columns and df['regime_rates'].notna().any())
    if has_regime:
        import matplotlib.patches as mpatches
        regime_col = df[['Date', 'regime_rates']].dropna()
        regime_col = regime_col.copy()
        regime_col['grp'] = (regime_col['regime_rates'] != regime_col['regime_rates'].shift()).cumsum()
        for _, grp in regime_col.groupby('grp'):
            regime = grp['regime_rates'].iloc[0]
            start  = grp['Date'].iloc[0]
            end    = grp['Date'].iloc[-1]
            color  = 'limegreen' if regime == 'T' else 'lightsalmon'
            ax_top.axvspan(start, end, alpha=0.12, color=color, linewidth=0)
        trending_patch = mpatches.Patch(color='limegreen',  alpha=0.4, label='추세장 (H>0.5)')
        ranging_patch  = mpatches.Patch(color='lightsalmon', alpha=0.4, label='횡보장 (H≤0.5)')
        regime_handles = [trending_patch, ranging_patch]
    else:
        regime_handles = []

    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_top2.get_legend_handles_labels()
    all_handles = lines1 + lines2 + regime_handles
    all_labels  = labels1 + labels2 + [h.get_label() for h in regime_handles]
    ax_top.legend(all_handles, all_labels, loc='upper left')

    # -----------------------------------------------------------------
    # Row 1: Individual Rates / Individual FX
    # -----------------------------------------------------------------
    for col in rate_cols:
        label = col.replace('_CumPnL', '')
        line = ax_ir.plot(df['Date'], df[col], label=label, alpha=0.8, linewidth=1.2)[0]
        ax_ir.text(last_date, df[col].iloc[-1], f" {df[col].iloc[-1]:.1f}",
                   color=line.get_color(), fontsize=7, va='center')
    ax_ir.set_ylabel('Individual Rates (bps)', fontweight='bold')
    ax_ir.set_title('Individual Rates Performance', fontsize=13)
    ax_ir.grid(True, linestyle='--', alpha=0.6)
    ax_ir.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='x-small')

    for col in fx_cols:
        label = col.replace('_CumPnL', '')
        line = ax_ifx.plot(df['Date'], df[col], label=label, alpha=0.8, linewidth=1.2)[0]
        ax_ifx.text(last_date, df[col].iloc[-1], f" {df[col].iloc[-1]:.2f}%",
                    color=line.get_color(), fontsize=7, va='center')
    ax_ifx.set_ylabel('Individual FX (%)', fontweight='bold')
    ax_ifx.set_title('Individual FX Performance', fontsize=13)
    ax_ifx.grid(True, linestyle='--', alpha=0.6)
    ax_ifx.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='x-small')

    # -----------------------------------------------------------------
    # Helper: plot strategy-type lines on an axis
    # -----------------------------------------------------------------
    def _plot_strategy_type(ax, col_map, unit_fmt, title):
        """col_map: dict of {label: column_name}"""
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan']
        missing = [c for c in col_map.values() if c not in df.columns]
        if missing:
            ax.text(0.5, 0.5, 'Data not available.\nRe-run backtest.',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11, color='gray')
            ax.set_title(title, fontsize=13)
            return

        for i, (label, col) in enumerate(col_map.items()):
            sharpe = calc_sharpe(df[col])
            color = colors[i % len(colors)]
            ax.plot(df['Date'], df[col], color=color, linewidth=2,
                    label=f'{label} (Sharpe: {sharpe:.2f})')
            last_val = df[col].iloc[-1]
            ax.text(last_date, last_val, f'  {unit_fmt.format(last_val)}',
                    color=color, fontsize=8, va='center')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.4)
        ax.set_title(title, fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left', fontsize='small')

    # -----------------------------------------------------------------
    # Row 2: Strategy-Type Rates / FX (MOM / MR / ADV)
    # -----------------------------------------------------------------
    if has_split:
        _plot_strategy_type(ax_sr, {
            'Momentum': 'mom_rates_cumpnl',
            'Mean-Rev': 'mr_rates_cumpnl',
            'Advanced': 'adv_rates_cumpnl',
        }, '{:.2f}', 'Strategy-Type: Rates (MOM / MR / ADV)')
        ax_sr.set_ylabel('Normalised Cum PnL (bps)', fontweight='bold')

        _plot_strategy_type(ax_sfx, {
            'Momentum': 'mom_fx_cumpnl',
            'Mean-Rev': 'mr_fx_cumpnl',
            'Advanced': 'adv_fx_cumpnl',
        }, '{:.2f}', 'Strategy-Type: FX (MOM / MR / ADV)')
        ax_sfx.set_ylabel('Normalised Cum PnL (%)', fontweight='bold')
    else:
        ax_sr.text(0.5, 0.5, 'MOM/MR/ADV columns not found.\nRe-run backtest.',
                   ha='center', va='center', transform=ax_sr.transAxes, fontsize=11, color='gray')
        ax_sr.set_title('Strategy-Type: Rates', fontsize=13)
        ax_sfx.text(0.5, 0.5, 'MOM/MR/ADV columns not found.\nRe-run backtest.',
                    ha='center', va='center', transform=ax_sfx.transAxes, fontsize=11, color='gray')
        ax_sfx.set_title('Strategy-Type: FX', fontsize=13)

    # -----------------------------------------------------------------
    # Row 3: Alpha Rates / Alpha FX (A1 / A2 / A3)
    # -----------------------------------------------------------------
    if has_alpha:
        _plot_strategy_type(ax_ar, {
            'Alpha1 (XSect)': 'a1_rates_cumpnl',
            'Alpha2 (Predict)': 'a2_rates_cumpnl',
            'Alpha3 (Regime)': 'a3_rates_cumpnl',
        }, '{:.2f}', 'Alpha Strategies: Rates (A1 / A2 / A3)')
        ax_ar.set_ylabel('Normalised Cum PnL (bps)', fontweight='bold')

        _plot_strategy_type(ax_afx, {
            'Alpha1 (XSect)': 'a1_fx_cumpnl',
            'Alpha2 (Predict)': 'a2_fx_cumpnl',
            'Alpha3 (Regime)': 'a3_fx_cumpnl',
        }, '{:.2f}', 'Alpha Strategies: FX (A1 / A2 / A3)')
        ax_afx.set_ylabel('Normalised Cum PnL (%)', fontweight='bold')

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    _end_date = end_date or df['Date'].iloc[-1].strftime('%Y-%m-%d')
    suffix_start = f"_{start_date}" if start_date else ""
    suffix_end   = f"_to{_end_date}"
    suffix_corr = f"_corr{max_correlation}" if max_correlation is not None else ""
    suffix_mode = f"_{mode}" if mode else ""
    suffix_sharpe = f"_rates{sharpe_rates:.2f}_fx{sharpe_fx:.2f}"
    output_path = f"trading_pnl{suffix_start}{suffix_end}{suffix_corr}{suffix_mode}{suffix_sharpe}.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Rates+FX PnL plot saved to {output_path}")

    # === Separate Index Plot ===
    if index_cols and 'total_index_cumpnl' in df.columns:
        sharpe_idx = calc_sharpe(df['total_index_cumpnl'])

        fig_idx, (ax_agg, ax_ind) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        # Aggregate index
        ax_agg.plot(df['Date'], df['total_index_cumpnl'], color='tab:green', linewidth=3,
                    label=f'Total Index (Sharpe: {sharpe_idx:.2f})')
        last_val_idx = df['total_index_cumpnl'].iloc[-1]
        ax_agg.text(last_date, last_val_idx, f" {last_val_idx:.2f}%", color='tab:green', fontweight='bold', va='center')
        ax_agg.set_ylabel('Total Index Cum PnL (%)', fontweight='bold')
        ax_agg.set_title(f'Index Strategy Performance{title_mode}', fontsize=16, pad=20)
        ax_agg.grid(True, linestyle='--', alpha=0.6)
        ax_agg.legend(loc='upper left')

        # Individual indices
        for col in index_cols:
            label = col.replace('_CumPnL', '')
            line = ax_ind.plot(df['Date'], df[col], label=label, alpha=0.8)[0]
            ax_ind.text(last_date, df[col].iloc[-1], f" {df[col].iloc[-1]:.2f}%", color=line.get_color(), fontsize=8, va='center')
        ax_ind.set_ylabel('Individual Indices (%)', fontweight='bold')
        ax_ind.set_title('Individual Index Performance', fontsize=14)
        ax_ind.grid(True, linestyle='--', alpha=0.6)
        ax_ind.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

        plt.xlabel('Date')
        fig_idx.tight_layout()

        index_output = f"trading_pnl_index{suffix_start}{suffix_end}{suffix_corr}{suffix_mode}_idx{sharpe_idx:.2f}.png"
        plt.savefig(index_output, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Index PnL plot saved to {index_output}")

if __name__ == "__main__":
    plot_pnl()
