"""
Portfolio Backtester for Global Macro Trading

Simulates monthly rebalancing and strategy selection based on performance and correlation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..factory.strategy_factory import StrategyFactory
from ..portfolio.selector import StrategySelector
from ..backtester.vectorbt_engine import VectorBTEngine
from ..backtester.batch_explorer import BatchExplorer

logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """Simulate portfolio performance with rebalancing."""
    
    def __init__(
        self,
        prices: pd.DataFrame,
        factory: StrategyFactory,
        selector_threshold: float = 0.9,
        max_correlation: float = 0.7,
        realistic: bool = False
    ):
        self.prices = prices
        self.factory = factory
        self.selector = StrategySelector(factory, sharpe_6m_threshold=selector_threshold)
        self.max_correlation = max_correlation
        self.realistic = realistic
        
    def run_simulation(
        self,
        start_date: str,
        end_date: str = None,
        freq: str = 'ME'
    ) -> Dict[str, Any]:
        """
        Run a walk-forward portfolio simulation.
        """
        all_dates = self.prices.index
        sim_start = pd.to_datetime(start_date)
        sim_end = pd.to_datetime(end_date) if end_date else all_dates[-1]
        
        # Generate rebalance dates
        rebalance_dates = pd.date_range(start=sim_start, end=sim_end, freq=freq)
        
        portfolio_returns = []
        history = []

        all_rets_df = None
        strat_configs = {}
        
        if self.realistic:
            logger.info("🕵️ Starting Realistic Discovery (Scanning all 4,500+ combinations)...")
            explorer = BatchExplorer(self.prices)
            all_rets_df, strat_configs = explorer.evaluate_all_strategies()
            logger.info(f"✅ Pre-calculated returns for {len(strat_configs)} strategies.")

        logger.info(f"🚀 Starting rebalancing simulation from {sim_start.date()} to {sim_end.date()}")

        for i in range(len(rebalance_dates) - 1):
            curr_date = rebalance_dates[i]
            next_date = rebalance_dates[i+1]
            
            logger.info(f"📅 Rebalancing at {curr_date.date()}...")
            
            qualified_strategies = []
            
            if self.realistic:
                # Realistic: Scan the pre-calculated returns up to curr_date
                slice_rets = all_rets_df[all_rets_df.index <= curr_date]
                if len(slice_rets) < 126:
                    continue
                
                # Performance thresholds from config
                # We use 1.0 (3Y) and 1.2 (6M)
                s3y_min = 1.0 # Or load from config
                s6m_min = 1.2
                
                # Vectorized Sharpe Calculation for ALL strategies
                means = slice_rets.mean() * 252
                stds = slice_rets.std() * np.sqrt(252)
                sharpes = means / stds.replace(0, np.nan)
                
                # 6M slice
                slice_6m = slice_rets.iloc[-126:]
                sharpe_6m = (slice_6m.mean() * 252) / (slice_6m.std() * np.sqrt(252)).replace(0, np.nan)
                
                # Filter strategies that meet both criteria
                qualified_ids = sharpes[(sharpes >= s3y_min) & (sharpe_6m >= s6m_min)].index
                
                for sid in qualified_ids:
                    config = strat_configs[sid].copy()
                    config['strategy_id'] = sid
                    config['performance'] = {
                        'sharpe_3y': sharpes[sid],
                        'sharpe_6m': sharpe_6m[sid]
                    }
                    qualified_strategies.append(config)
            else:
                # Biased: Only use strategies currently in factory
                prices_until_now = self.prices[self.prices.index <= curr_date]
                if len(prices_until_now) < 126:
                    continue
                
                backtester = VectorBTEngine(prices_until_now)
                candidate_strategies = self.factory.filter_by_sharpe_3y(0.0) 
                
                for strategy in candidate_strategies:
                    res = backtester.run_backtest(strategy)
                    if res.sharpe_6m >= self.selector.sharpe_threshold:
                        strat_copy = strategy.copy()
                        strat_copy['performance'] = res.to_dict()
                        qualified_strategies.append(strat_copy)

            if not qualified_strategies:
                continue
            
            # 2. Correlation filter
            prices_until_now = self.prices[self.prices.index <= curr_date]
            active_strategies = self.selector._filter_by_correlation(qualified_strategies, prices_until_now, self.max_correlation)
            
            if not active_strategies:
                continue
                
            logger.info(f"   Picked {len(active_strategies)} diversified strategies.")

            # 3. Next period returns
            prices_next_period = self.prices[(self.prices.index > curr_date) & (self.prices.index <= next_date)]
            if prices_next_period.empty:
                continue
                
            period_returns = []
            for strategy in active_strategies:
                if self.realistic:
                    # Just pluck from pre-calculated DF
                    strat_rets = all_rets_df.loc[prices_next_period.index, strategy['strategy_id']]
                else:
                    full_slice = self.prices[self.prices.index <= next_date]
                    res = VectorBTEngine(full_slice).run_backtest(strategy)
                    strat_rets = res.returns[res.returns.index > curr_date]
                
                period_returns.append(strat_rets)
            
            if period_returns:
                combined_period = pd.concat(period_returns, axis=1).mean(axis=1)
                portfolio_returns.append(combined_period)
                history.append({
                    'rebalance_date': curr_date,
                    'num_strategies': len(active_strategies),
                    'strategies': [s.get('strategy_id', s.get('id', 'unknown')) for s in active_strategies]
                })

        if not portfolio_returns:
            return {'error': 'No returns generated'}
            
        all_returns = pd.concat(portfolio_returns)
        equity_curve = (1 + all_returns).cumprod()
        total_return = equity_curve.iloc[-1] - 1
        sharpe = (all_returns.mean() / all_returns.std()) * np.sqrt(252) if all_returns.std() != 0 else 0
        mdd = (equity_curve / equity_curve.cummax() - 1).min()
        
        results = {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': mdd,
            'returns': all_returns,
            'equity_curve': equity_curve,
            'history': history
        }
        
        self.plot_results(results)
        return results

    def plot_results(self, results: Dict[str, Any]):
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'], label=f"Portfolio (Sharpe: {results['sharpe']:.2f})")
        plt.title(f"Portfolio {'Realistic' if self.realistic else 'Index'} Backtest (Max Corr: {self.max_correlation})")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("portfolio_backtest.png")
        logger.info(f"📊 Saved {'realistic' if self.realistic else 'index'} backtest plot.")
