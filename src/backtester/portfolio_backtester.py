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

logger = logging.getLogger(__name__)

class PortfolioBacktester:
    """Simulate portfolio performance with rebalancing."""
    
    def __init__(
        self,
        prices: pd.DataFrame,
        factory: StrategyFactory,
        selector_threshold: float = 0.9,
        max_correlation: float = 0.7
    ):
        self.prices = prices
        self.factory = factory
        self.selector = StrategySelector(factory, sharpe_6m_threshold=selector_threshold)
        self.max_correlation = max_correlation
        
    def run_simulation(
        self,
        start_date: str,
        end_date: str = None,
        freq: str = 'ME'
    ) -> Dict[str, Any]:
        """
        Run a walk-forward portfolio simulation.
        
        Args:
            start_date: Simulation start date
            end_date: Simulation end date
            freq: Rebalance frequency (M=Month end, W=Week end)
            
        Returns:
            Simulation results
        """
        all_dates = self.prices.index
        sim_start = pd.to_datetime(start_date)
        sim_end = pd.to_datetime(end_date) if end_date else all_dates[-1]
        
        # Generate rebalance dates
        rebalance_dates = pd.date_range(start=sim_start, end=sim_end, freq=freq)
        
        portfolio_returns = []
        portfolio_weights = []
        history = []

        logger.info(f"🚀 Starting rebalancing simulation from {sim_start.date()} to {sim_end.date()}")

        for i in range(len(rebalance_dates) - 1):
            curr_date = rebalance_dates[i]
            next_date = rebalance_dates[i+1]
            
            logger.info(f"📅 Rebalancing at {curr_date.date()}...")
            
            # 1. Evaluate ALL strategies in factory based on data up to curr_date
            prices_until_now = self.prices[self.prices.index <= curr_date]
            if len(prices_until_now) < 126: # Need at least 6 months
                continue
                
            qualified_strategies = []
            backtester = VectorBTEngine(prices_until_now)
            
            # Get all strategies that pass the base 3y Sharpe from index (pre-filter)
            candidate_strategies = self.factory.filter_by_sharpe_3y(0.0) # All stored
            
            for strategy in candidate_strategies:
                res = backtester.run_backtest(strategy)
                # Calculate 6m Sharpe on historical data slice
                if res.sharpe_6m >= self.selector.sharpe_threshold:
                    # Temporary update performance for the correlation filter to see
                    # We store it in a copy to avoid corrupting the real strategy dicts
                    strat_copy = strategy.copy()
                    strat_copy['performance'] = res.to_dict()
                    qualified_strategies.append(strat_copy)

            if not qualified_strategies:
                logger.warning(f"⚠️ No strategies passed Sharpe threshold at {curr_date.date()}")
                continue
            
            # 2. Apply correlation filter on qualified strategies
            # We bypass _refresh_sync because we have our own performance metrics now
            active_strategies = self.selector._filter_by_correlation(qualified_strategies, prices_until_now, self.max_correlation)
            
            if not active_strategies:
                logger.warning(f"⚠️ No strategies selected after corr filter at {curr_date.date()}")
                continue
                
            logger.info(f"   Picked {len(active_strategies)} diversified strategies.")

            # 3. Get returns for the next period (Out-of-Sample)
            prices_next_period = self.prices[(self.prices.index > curr_date) & (self.prices.index <= next_date)]
            if prices_next_period.empty:
                continue
                
            # Calculate returns for this period for chosen strategies
            period_returns = []
            
            for strategy in active_strategies:
                # Run backtest on the slice that includes the next period
                full_slice = self.prices[self.prices.index <= next_date]
                res = VectorBTEngine(full_slice).run_backtest(strategy)
                
                # Extract only the returns for the next period
                strat_rets = res.returns[res.returns.index > curr_date]
                period_returns.append(strat_rets)
            
            # 4. Aggregate period returns (Equal weighted)
            if period_returns:
                combined_period = pd.concat(period_returns, axis=1).mean(axis=1)
                portfolio_returns.append(combined_period)
                
                history.append({
                    'rebalance_date': curr_date,
                    'num_strategies': len(active_strategies),
                    'strategies': [s['strategy_id'] for s in active_strategies]
                })

        if not portfolio_returns:
            return {'error': 'No returns generated'}
            
        # Final Equity Curve
        all_returns = pd.concat(portfolio_returns)
        equity_curve = (1 + all_returns).cumprod()
        
        # Calculate Stats
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
        """Plot the equity curve."""
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'], label=f"Portfolio (Sharpe: {results['sharpe']:.2f})")
        plt.title(f"Portfolio Rebalancing Performance (Max Corr: {self.max_correlation})")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig("portfolio_backtest.png")
        logger.info("📊 Saved backtest plot for portfolio_backtest.png")
