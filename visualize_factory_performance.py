"""
Visualize Overall Performance of Strategy Factory
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.backtester.vectorbt_engine import VectorBTEngine
from src.factory.strategy_factory import StrategyFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_factory():
    # 1. Initialize
    factory = StrategyFactory()
    data_loader = DataLoader()
    
    # 2. Load Data (Last 3+ years)
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=3, months=3)).strftime('%Y-%m-%d')
    logger.info(f"📊 Loading data since {start_date}...")
    prices_raw = data_loader.load_data(start_date=start_date)
    preprocessor = DataPreprocessor(prices_raw)
    prices = preprocessor.clean().get_data()
    
    # 3. Get Strategies
    strategies = factory.filter_by_sharpe_3y(0.0) # Load all
    logger.info(f"🏢 Total strategies in factory: {len(strategies)}")
    
    if not strategies:
        logger.warning("No strategies found in factory.")
        return

    # 4. Run Backtests and Gather Equity Curves
    engine = VectorBTEngine(prices)
    all_cum_returns = {}
    sharpe_ratios = []
    
    logger.info("🧪 Running backtests for all strategies...")
    for strat in tqdm(strategies):
        try:
            # We want the cumulative returns series
            # VectorBT engine's run_backtest usually returns BacktestResult which has metrics
            # To get the series, we might need to peek into the engine logic or modify it
            # For now, let's use a simplified calculation similar to visualize_strategy.py
            asset = strat['asset']
            if asset not in prices.columns:
                continue
                
            # We'll use the result to get the sharpe
            res = engine.run_backtest(strat, start_date=start_date)
            sharpe_ratios.append(res.sharpe_3y)
            
            # Re-calculating cumulative returns for plotting
            # (In a real system, VectorBTEngine should return the series)
            from src.strategies.momentum import MomentumStrategy
            from src.strategies.mean_reversion import MeanReversionStrategy
            
            if strat['strategy_type'] == 'momentum':
                entries, exits = MomentumStrategy.generate_signals(
                    prices, asset, strat['strategy_name'], strat['params'], strat.get('related_asset')
                )
                pos = pd.Series(0, index=prices.index)
                curr = 0
                for i in range(len(entries)):
                    if entries.iloc[i] and curr == 0: curr = 1
                    elif exits.iloc[i] and curr == 1: curr = 0
                    pos.iloc[i] = curr
            else:
                l_e, l_x, s_e, s_x = MeanReversionStrategy.generate_signals(
                    prices, asset, strat['strategy_name'], strat['params'], strat.get('related_asset')
                )
                pos = pd.Series(0, index=prices.index)
                curr = 0
                for i in range(len(l_e)):
                    if l_e.iloc[i] and curr == 0: curr = 1
                    elif s_e.iloc[i] and curr == 0: curr = -1
                    elif l_x.iloc[i] and curr == 1: curr = 0
                    elif s_x.iloc[i] and curr == -1: curr = 0
                    pos.iloc[i] = curr
            
            # Use log prices or original prices for returns? 
            # Consistent with engine, but let's use pct_change on prices
            # Handle non-positive prices by shifting if necessary (as in fix)
            p_series = prices[asset]
            if (p_series <= 0).any():
                p_series = np.exp(p_series / 100.0) * 100.0
            
            ret = p_series.pct_change() * pos.shift(1)
            cum_ret = (1 + ret.fillna(0)).cumprod()
            all_cum_returns[strat['strategy_id']] = cum_ret
            
        except Exception as e:
            logger.error(f"Error processing {strat['strategy_id']}: {e}")

    # 5. Plotting
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left Plot: Equity Curves
    if all_cum_returns:
        curves_df = pd.DataFrame(all_cum_returns)
        
        # Filter for 3Y (last 756 business days approx)
        curves_df = curves_df.iloc[-756:]
        # Re-base to 1.0 at the start of original 3Y window
        curves_df = curves_df / curves_df.iloc[0]
        
        for col in curves_df.columns:
            ax1.plot(curves_df.index, curves_df[col], color='gray', alpha=0.1, linewidth=0.5)
        
        # Plot Average
        avg_curve = curves_df.mean(axis=1)
        ax1.plot(curves_df.index, avg_curve, color='blue', linewidth=3, label='Equally Weighted Portfolio (Avg)')
        
        # Plot Top 3 by Sharpe 3Y
        top_ids = sorted(strategies, key=lambda x: x['performance']['sharpe_3y'], reverse=True)[:3]
        colors = ['green', 'orange', 'purple']
        for i, s in enumerate(top_ids):
            if s['strategy_id'] in curves_df.columns:
                ax1.plot(curves_df.index, curves_df[s['strategy_id']], color=colors[i], linewidth=1.5, 
                         label=f"Top {i+1}: {s['strategy_id']} (Sharpe: {s['performance']['sharpe_3y']:.2f})")

        ax1.set_title("Cumulative Returns (3Y) - All Strategies", fontsize=15)
        ax1.set_ylabel("Growth of $1")
        ax1.legend(loc='upper left')
    
    # Right Plot: Sharpe Distribution
    if sharpe_ratios:
        sns.histplot(sharpe_ratios, bins=20, kde=True, ax=ax2, color='skyblue')
        ax2.axvline(np.mean(sharpe_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(sharpe_ratios):.2f}')
        ax2.set_title("Distribution of Sharpe Ratios (3Y)", fontsize=15)
        ax2.set_xlabel("Sharpe Ratio")
        ax2.legend()

    plt.tight_layout()
    output_path = Path("factory_performance_3y.png")
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"✅ Visualization saved to {output_path.absolute()}")
    print(f"\n📈 Factory Performance Summary:")
    print(f"   Total Strategies: {len(all_cum_returns)}")
    print(f"   Avg Sharpe (3Y): {np.mean(sharpe_ratios):.2f}")
    print(f"   Best Sharpe (3Y): {max(sharpe_ratios):.2f}")
    print(f"   Image saved: {output_path.name}")

if __name__ == "__main__":
    visualize_factory()
