"""
Strategy Visualization and Reporting Tool

Loads a saved strategy, runs backtest, generates plots,
and creates a Markdown report for Obsidian.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.backtester.vectorbt_engine import VectorBTEngine, BacktestResult
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyVisualizer:
    def __init__(self, obsidian_path: str):
        self.obsidian_path = Path(obsidian_path)
        self.obsidian_path.mkdir(parents=True, exist_ok=True)
        self.data_loader = DataLoader()
        self.factory_dir = Path(__file__).parent / 'src' / 'factory' / 'strategies'
        
    def load_strategy(self, strategy_id: str) -> dict:
        strategy_file = self.factory_dir / f"{strategy_id}.json"
        if not strategy_file.exists():
            raise FileNotFoundError(f"Strategy {strategy_id} not found at {strategy_file}")
        with open(strategy_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def run_analysis(self, strategy_id: str):
        # 1. Load Strategy
        strategy = self.load_strategy(strategy_id)
        asset = strategy['asset']
        logger.info(f"📈 Analyzing Strategy: {strategy_id} for {asset}")
        
        # 2. Load Data
        prices = self.data_loader.load_data(start_date="2020-01-01")
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()
        
        # 3. Generate Signals
        if strategy['strategy_type'] == 'momentum':
            entries, exits = MomentumStrategy.generate_signals(
                prices, asset, strategy['strategy_name'], strategy['params'], strategy.get('related_asset')
            )
            # Long-only for momentum
            long_entries, long_exits = entries, exits
            short_entries = pd.Series(False, index=prices.index)
            short_exits = pd.Series(False, index=prices.index)
        else:
            long_entries, long_exits, short_entries, short_exits = MeanReversionStrategy.generate_signals(
                prices, asset, strategy['strategy_name'], strategy['params'], strategy.get('related_asset')
            )
            
        # 4. Run Backtest
        engine = VectorBTEngine(prices)
        result = engine.run_backtest(strategy)
        
        # 5. Create Plots
        self._create_plots(prices[asset], long_entries, long_exits, short_entries, short_exits, strategy_id)
        
        # 6. Generate Markdown Report
        self._generate_report(strategy, result)
        
    def _create_plots(self, prices, l_entries, l_exits, s_entries, s_exits, strategy_id):
        sns.set_theme(style="darkgrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Price and Signals
        ax1.plot(prices.index, prices.values, label='Price', color='black', alpha=0.6)
        
        # Entries/Exits point plots
        if l_entries.any():
            ax1.scatter(prices.index[l_entries], prices[l_entries], marker='^', color='green', label='Long Entry', s=100)
        if l_exits.any():
            ax1.scatter(prices.index[l_exits], prices[l_exits], marker='v', color='red', label='Long Exit', s=100)
        if s_entries.any():
            ax1.scatter(prices.index[s_entries], prices[s_entries], marker='v', color='orange', label='Short Entry', s=100)
        if s_exits.any():
            ax1.scatter(prices.index[s_exits], prices[s_exits], marker='^', color='blue', label='Short Exit', s=100)
            
        ax1.set_title(f"Price Action and Trading Signals: {strategy_id}", fontsize=16)
        ax1.legend()
        
        # Plot 2: Cumulative Returns (Simplified Simulation for Plot)
        # Detailed returns should come from vectorbt if available, but for simplicity:
        position = pd.Series(0, index=prices.index)
        current = 0
        for i in range(len(prices)):
            if l_entries.iloc[i]: current = 1
            elif s_entries.iloc[i]: current = -1
            elif l_exits.iloc[i] and current == 1: current = 0
            elif s_exits.iloc[i] and current == -1: current = 0
            position.iloc[i] = current
            
        strat_returns = prices.pct_change() * position.shift(1)
        cum_returns = (1 + strat_returns.fillna(0)).cumprod()
        
        ax2.plot(cum_returns.index, cum_returns.values, label='Cumulative Returns', color='blue')
        ax2.fill_between(cum_returns.index, 1, cum_returns.values, where=(cum_returns.values >= 1), color='green', alpha=0.1)
        ax2.fill_between(cum_returns.index, 1, cum_returns.values, where=(cum_returns.values < 1), color='red', alpha=0.1)
        ax2.set_title("Equity Curve", fontsize=16)
        ax2.legend()
        
        plt.tight_layout()
        img_path = self.obsidian_path / f"{strategy_id}_performance.png"
        plt.savefig(img_path)
        plt.close()
        logger.info(f"✅ Performance plot saved to {img_path}")

    def _generate_report(self, strategy, result: BacktestResult):
        strategy_id = strategy['strategy_id']
        report_path = self.obsidian_path / f"{strategy_id}_Report.md"
        
        md_content = f"""# Strategy Report: {strategy_id}

## 📄 Strategy Overview
- **Asset**: {strategy['asset']}
- **Type**: {strategy['strategy_type']}
- **Name**: {strategy['strategy_name']}
- **Created**: {strategy['created_at']}
- **Related Asset**: {strategy['related_asset'] or 'None'}

### ⚙️ Parameters
```json
{json.dumps(strategy['params'], indent=2)}
```

## 📊 Performance Indicators
| Metric | Value |
| :--- | :--- |
| **Sharpe Ratio (3Y)** | {result.sharpe_3y:.2f} |
| **Sharpe Ratio (6M)** | {result.sharpe_6m:.2f} |
| **Total Return** | {result.total_return*100:.2f}% |
| **Annualized Return** | {result.annual_return*100:.2f}% |
| **Max Drawdown** | {result.max_drawdown*100:.2f}% |
| **Win Rate** | {result.win_rate*100:.2f}% |
| **Number of Trades** | {result.num_trades} |
| **Calmar Ratio** | {result.calmar_ratio:.2f} |

## 📉 Performance Visualization
![[{strategy_id}_performance.png]]

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        logger.info(f"✅ Obsidian report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize strategy performance and generate Obsidian report.')
    parser.add_argument('--strategy-id', required=True, help='ID of the strategy to visualize')
    parser.add_argument('--obsidian-dir', default=r'D:\★사용자 폴더\Documents\Obsidian\sondo\Strategy_Report', help='Obsidian report directory')
    
    args = parser.parse_args()
    
    visualizer = StrategyVisualizer(args.obsidian_dir)
    visualizer.run_analysis(args.strategy_id)
