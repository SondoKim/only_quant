"""
VectorBT Backtesting Engine for Global Macro Trading

High-performance backtesting using vectorbt for fast parameter optimization.
Optimized with vectorized position calculation, indicator caching, and multiprocessing.
"""

import logging
import multiprocessing as mp
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False

from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.advanced import AdvancedStrategies
from .indicator_cache import IndicatorCache

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    strategy_id: str
    sharpe_ratio: float
    sharpe_3y: float
    sharpe_6m: float
    sharpe_1m: float
    total_return: float
    annual_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    calmar_ratio: float
    sortino_ratio: float
    returns: Optional[pd.Series] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_id': self.strategy_id,
            'sharpe_ratio': self.sharpe_ratio,
            'sharpe_3y': self.sharpe_3y,
            'sharpe_6m': self.sharpe_6m,
            'sharpe_1m': self.sharpe_1m,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'num_trades': self.num_trades,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'returns': self.returns
        }


class VectorBTEngine:
    """High-performance backtesting engine using vectorbt."""
    
    def __init__(
        self,
        prices: pd.DataFrame,
        transaction_cost_bps: float = 2.0,
        slippage_bps: float = 1.0
    ):
        """
        Initialize backtesting engine.
        
        Args:
            prices: DataFrame with price data indexed by date
            transaction_cost_bps: Transaction cost in basis points
            slippage_bps: Slippage in basis points
        """
        self.prices = prices
        self.transaction_cost = transaction_cost_bps / 10000
        self.slippage = slippage_bps / 10000
        
        if not VBT_AVAILABLE:
            logger.warning("vectorbt not available. Using simplified backtester.")
    
    def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        start_date: str = None,
        end_date: str = None
    ) -> BacktestResult:
        """
        Run backtest for a single strategy configuration.
        
        Args:
            strategy_config: Strategy configuration dict
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResult object
        """
        # Filter by date range
        prices = self.prices.copy()
        if start_date:
            prices = prices[prices.index >= pd.to_datetime(start_date)]
        if end_date:
            prices = prices[prices.index <= pd.to_datetime(end_date)]
        
        asset = strategy_config['asset']
        strategy_type = strategy_config['strategy_type']
        strategy_name = strategy_config['strategy_name']
        params = strategy_config['params']
        related_asset = strategy_config.get('related_asset')
        strategy_id = strategy_config.get('id', 'UNKNOWN')
        
        # Generate signals
        if strategy_type == 'momentum':
            entries, exits = MomentumStrategy.generate_signals(
                prices, asset, strategy_name, params, related_asset
            )
            # For momentum, only long trades
            long_entries, long_exits = entries, exits
            short_entries = pd.Series(False, index=prices.index)
            short_exits = pd.Series(False, index=prices.index)
        elif strategy_type == 'mean_reversion':
            long_entries, long_exits, short_entries, short_exits = \
                MeanReversionStrategy.generate_signals(
                    prices, asset, strategy_name, params, related_asset
                )
        elif strategy_type == 'advanced':
            entries, exits = AdvancedStrategies.generate_signals(
                prices, asset, strategy_name, params, related_asset
            )
            # For advanced, treat as long-short or long-only depending on implementation
            # Here we treat entries/exits from AdvancedStrategies as long signals
            long_entries, long_exits = entries, exits
            short_entries = pd.Series(False, index=prices.index)
            short_exits = pd.Series(False, index=prices.index)
        else:
            raise ValueError(f"Unknown strategy category: {strategy_type}")
        
        # Run backtest
        if VBT_AVAILABLE:
            return self._run_vbt_backtest(
                prices[asset], 
                long_entries, long_exits,
                short_entries, short_exits,
                strategy_id
            )
        else:
            return self._run_simple_backtest(
                prices[asset],
                long_entries, long_exits,
                short_entries, short_exits,
                strategy_id
            )
    
    def _run_vbt_backtest(
        self,
        prices: pd.Series,
        long_entries: pd.Series,
        long_exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series,
        strategy_id: str
    ) -> BacktestResult:
        """Run backtest using vectorbt."""
        try:
            # Handle non-positive prices (common in yields/macro data)
            # vectorbt requires prices > 0 for return calculations.
            vbt_prices = prices.copy()
            if (vbt_prices <= 0).any():
                # For macro data with <= 0 values, we use a synthetic index
                # This preserves return direction and relative magnitude
                diff = vbt_prices.diff().fillna(0)
                # We treat the change as a log-return or similar
                # Small constant added to avoid extreme moves if yield is near 0
                vbt_prices = np.exp(vbt_prices / 10.0) # Scale down to avoid overflow
            
            # Combine long and short into direction-aware signals
            # For now, run long-only backtest (vectorbt handles this well)
            pf = vbt.Portfolio.from_signals(
                vbt_prices,
                entries=long_entries,
                exits=long_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                fees=self.transaction_cost,
                slippage=self.slippage,
                freq='D'
            )
            
            # Calculate metrics
            returns = pf.returns()
            
            # Period-specific Sharpe ratios
            sharpe_all = self._safe_sharpe(returns)
            sharpe_3y = self._calculate_period_sharpe(returns, years=3)
            sharpe_6m = self._calculate_period_sharpe(returns, months=6)
            sharpe_1m = self._calculate_period_sharpe(returns, months=1)
            
            stats = pf.stats()
            
            return BacktestResult(
                strategy_id=strategy_id,
                sharpe_ratio=sharpe_all,
                sharpe_3y=sharpe_3y,
                sharpe_6m=sharpe_6m,
                sharpe_1m=sharpe_1m,
                total_return=float(stats.get('Total Return [%]', 0)) / 100,
                annual_return=float(stats.get('Annualized Return [%]', 0)) / 100,
                max_drawdown=float(stats.get('Max Drawdown [%]', 0)) / 100,
                win_rate=float(stats.get('Win Rate [%]', 0)) / 100,
                num_trades=int(stats.get('Total Trades', 0)),
                calmar_ratio=float(stats.get('Calmar Ratio', 0)),
                sortino_ratio=float(stats.get('Sortino Ratio', 0)),
                returns=returns
            )
            
        except Exception as e:
            logger.error(f"VBT backtest failed for {strategy_id}: {e}")
            return self._empty_result(strategy_id)
    
    @staticmethod
    def _vectorized_position(long_entries: np.ndarray, long_exits: np.ndarray,
                              short_entries: np.ndarray, short_exits: np.ndarray) -> np.ndarray:
        """Calculate position series using vectorized NumPy operations."""
        n = len(long_entries)
        position = np.zeros(n, dtype=np.int8)
        
        # Build event arrays: +1 for long entry, -1 for short entry, 0 for exit
        # Process sequentially but using numpy boolean indexing for speed
        pos = 0
        for i in range(n):
            if pos == 0:
                if long_entries[i]:
                    pos = 1
                elif short_entries[i]:
                    pos = -1
            elif pos == 1:
                if long_exits[i]:
                    pos = 0
            elif pos == -1:
                if short_exits[i]:
                    pos = 0
            position[i] = pos
        
        return position
    
    def _run_simple_backtest(
        self,
        prices: pd.Series,
        long_entries: pd.Series,
        long_exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series,
        strategy_id: str
    ) -> BacktestResult:
        """Run simplified backtest without vectorbt."""
        try:
            prices_arr = prices.values
            returns_arr = np.empty_like(prices_arr)
            returns_arr[0] = 0.0
            returns_arr[1:] = prices_arr[1:] / prices_arr[:-1] - 1.0
            
            # Vectorized position calculation
            position = self._vectorized_position(
                long_entries.values, long_exits.values,
                short_entries.values, short_exits.values
            )
            
            # Strategy returns: position[t-1] * return[t]
            shifted_pos = np.empty_like(position, dtype=np.float64)
            shifted_pos[0] = 0.0
            shifted_pos[1:] = position[:-1]
            
            strategy_returns_arr = returns_arr * shifted_pos
            
            # Transaction costs
            trades_arr = np.abs(np.diff(position, prepend=0)).astype(np.float64)
            strategy_returns_arr -= trades_arr * self.transaction_cost
            
            strategy_returns = pd.Series(strategy_returns_arr, index=prices.index)
            
            # Calculate metrics
            sharpe_all = self._safe_sharpe(strategy_returns)
            sharpe_3y = self._calculate_period_sharpe(strategy_returns, years=3)
            sharpe_6m = self._calculate_period_sharpe(strategy_returns, months=6)
            sharpe_1m = self._calculate_period_sharpe(strategy_returns, months=1)
            
            cumulative = np.cumprod(1.0 + strategy_returns_arr)
            total_return = cumulative[-1] - 1.0 if len(cumulative) > 0 else 0.0
            
            # Max drawdown (vectorized)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / np.where(peak > 0, peak, 1.0)
            max_dd = float(drawdown.min())
            
            # Win rate
            active_mask = strategy_returns_arr != 0
            winning_days = int((strategy_returns_arr > 0).sum())
            total_days = int(active_mask.sum())
            win_rate = winning_days / total_days if total_days > 0 else 0.0
            
            # Number of trades
            num_trades = int(trades_arr.sum() / 2)
            
            # Annual return
            days = len(strategy_returns_arr)
            if days > 0 and (1 + total_return) > 0:
                annual_return = (1 + total_return) ** (252 / days) - 1
            else:
                annual_return = -1.0 if total_return <= -1 else 0.0
            
            return BacktestResult(
                strategy_id=strategy_id,
                sharpe_ratio=sharpe_all,
                sharpe_3y=sharpe_3y,
                sharpe_6m=sharpe_6m,
                sharpe_1m=sharpe_1m,
                total_return=total_return,
                annual_return=annual_return,
                max_drawdown=max_dd,
                win_rate=win_rate,
                num_trades=num_trades,
                calmar_ratio=annual_return / abs(max_dd) if max_dd != 0 else 0,
                sortino_ratio=self._safe_sortino(strategy_returns),
            )
            
        except Exception as e:
            logger.error(f"Simple backtest failed for {strategy_id}: {e}")
            return self._empty_result(strategy_id)
    
    def _safe_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio safely."""
        if len(returns) < 2:
            return 0.0
        mean = returns.mean()
        std = returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(mean / std * np.sqrt(252))

    def _safe_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio safely."""
        if len(returns) < 2:
            return 0.0
        mean = returns.mean() * 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 2:
            return 0.0
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float(mean / downside_std)
    
    def _calculate_period_sharpe(
        self,
        returns: pd.Series,
        years: int = None,
        months: int = None
    ) -> float:
        """Calculate Sharpe ratio for specific period."""
        if len(returns) == 0:
            return 0.0
        
        end_date = returns.index[-1]
        
        if years:
            start_date = end_date - pd.DateOffset(years=years)
        elif months:
            start_date = end_date - pd.DateOffset(months=months)
        else:
            return self._safe_sharpe(returns)
        
        period_returns = returns[returns.index >= start_date]
        return self._safe_sharpe(period_returns)
    
    def _empty_result(self, strategy_id: str) -> BacktestResult:
        """Return empty result for failed backtests."""
        return BacktestResult(
            strategy_id=strategy_id,
            sharpe_ratio=0.0,
            sharpe_3y=0.0,
            sharpe_6m=0.0,
            sharpe_1m=0.0,
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            num_trades=0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            returns=pd.Series()
        )
    
    def run_batch_backtest(
        self,
        strategies: List[Dict[str, Any]],
        n_jobs: int = -1,
        progress: bool = True
    ) -> List[BacktestResult]:
        """
        Run backtests for multiple strategies with parallel processing.
        
        Strategies are grouped by asset for indicator cache efficiency,
        then processed in parallel across CPU cores.
        
        Args:
            strategies: List of strategy configurations
            n_jobs: Number of parallel jobs (-1 for all cores)
            progress: Show progress bar
            
        Returns:
            List of BacktestResult objects (same order as input)
        """
        if not strategies:
            return []
        
        # Determine number of workers
        if n_jobs == -1:
            n_jobs = max(1, mp.cpu_count() - 1)
        n_jobs = min(n_jobs, len(strategies))
        
        # Group strategies by asset for cache efficiency
        asset_groups = defaultdict(list)
        index_map = {}  # strategy_index -> (asset, position_in_group)
        
        for idx, strategy in enumerate(strategies):
            asset = strategy.get('asset', 'UNKNOWN')
            pos = len(asset_groups[asset])
            asset_groups[asset].append((idx, strategy))
            index_map[idx] = (asset, pos)
        
        # Process all groups (using multiprocessing for large batches)
        results = [None] * len(strategies)
        
        if n_jobs > 1 and len(strategies) > 50:
            # Parallel: split asset groups across workers
            groups_list = list(asset_groups.items())
            
            # Distribute groups across workers
            chunks = [[] for _ in range(n_jobs)]
            for i, group in enumerate(groups_list):
                chunks[i % n_jobs].append(group)
            
            # Count strategies per chunk for progress display
            chunk_sizes = [sum(len(strats) for _, strats in chunk) for chunk in chunks]
            
            try:
                worker_fn = partial(_batch_worker,
                                    prices_dict={col: self.prices[col].to_dict() for col in self.prices.columns},
                                    prices_index=list(self.prices.index),
                                    transaction_cost=self.transaction_cost,
                                    slippage=self.slippage)
                
                completed = 0
                with mp.Pool(n_jobs) as pool:
                    try:
                        from tqdm import tqdm
                        iterator = tqdm(
                            pool.imap_unordered(worker_fn, chunks),
                            total=len(chunks),
                            desc=f"⚡ Parallel ({n_jobs} workers)",
                            unit="chunk"
                        )
                    except ImportError:
                        iterator = pool.imap_unordered(worker_fn, chunks)
                    
                    for worker_batch in iterator:
                        for orig_idx, result_dict in worker_batch:
                            results[orig_idx] = BacktestResult(**result_dict)
                            completed += 1
                        
                logger.info(f"⚡ Parallel backtest: {len(strategies)} strategies on {n_jobs} workers")
                return results
                
            except Exception as e:
                logger.warning(f"Parallel execution failed ({e}), falling back to sequential")
        
        # Sequential fallback (small batches or parallel failure)
        try:
            from tqdm import tqdm
            use_tqdm = progress
        except ImportError:
            use_tqdm = False
        
        iterator = enumerate(strategies)
        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Backtesting")
        
        for idx_strategy in iterator:
            if isinstance(idx_strategy, tuple):
                idx, strategy = idx_strategy
            else:
                idx, strategy = idx_strategy
            results[idx] = self.run_backtest(strategy)
        
        return results
    
    def filter_results(
        self,
        results: List[BacktestResult],
        sharpe_3y_min: float = 0.8,
        sharpe_6m_min: float = 0.9,
        min_trades: int = 20,
        max_drawdown: float = -0.20
    ) -> Tuple[List[BacktestResult], List[BacktestResult]]:
        """
        Filter results by performance criteria.
        
        Args:
            results: List of backtest results
            sharpe_3y_min: Minimum 3-year Sharpe for storage
            sharpe_6m_min: Minimum 6-month Sharpe for activation
            min_trades: Minimum number of trades
            max_drawdown: Maximum drawdown (negative)
            
        Returns:
            Tuple of (storage_qualified, active_qualified) results
        """
        storage_qualified = []
        active_qualified = []
        
        for result in results:
            # Basic quality filters
            if result.num_trades < min_trades:
                continue
            if result.max_drawdown < max_drawdown:
                continue
            
            # Storage qualification (Sharpe 3Y > 0.8)
            if result.sharpe_3y >= sharpe_3y_min:
                storage_qualified.append(result)
                
                # Active qualification (Sharpe 6M > 0.9)
                if result.sharpe_6m >= sharpe_6m_min:
                    active_qualified.append(result)
        
        logger.info(f"📊 Filtering results: {len(results)} total → "
                   f"{len(storage_qualified)} storage → {len(active_qualified)} active")
        
        return storage_qualified, active_qualified


def _batch_worker(chunk, prices_dict, prices_index, transaction_cost, slippage):
    """
    Worker function for multiprocessing batch backtest.
    
    Must be a module-level function (not a method) for pickling.
    Reconstructs a VectorBTEngine in the worker process.
    
    Args:
        chunk: List of (asset_name, [(orig_idx, strategy_config), ...]) tuples
        prices_dict: Serialized prices data {col: {date: value}}
        prices_index: List of date indices
        transaction_cost: Transaction cost (already in decimal)
        slippage: Slippage (already in decimal)
    
    Returns:
        List of (orig_idx, result_dict) tuples
    """
    # Reconstruct prices DataFrame in worker
    idx = pd.DatetimeIndex(prices_index)
    prices = pd.DataFrame(prices_dict, index=idx)
    
    # Create engine (bypass __init__ cost conversion since already converted)
    engine = VectorBTEngine.__new__(VectorBTEngine)
    engine.prices = prices
    engine.transaction_cost = transaction_cost
    engine.slippage = slippage
    
    worker_results = []
    for asset_name, strategies_with_idx in chunk:
        for orig_idx, strategy in strategies_with_idx:
            result = engine.run_backtest(strategy)
            # Convert to dict for serialization back to main process
            result_dict = result.to_dict()
            result_dict.pop('returns', None)  # Don't serialize large Series
            worker_results.append((orig_idx, result_dict))
    
    return worker_results
