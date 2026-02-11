"""
Portfolio Strategy Selector for Global Macro Trading

Selects active strategies and generates aggregated trading signals.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..factory.strategy_factory import StrategyFactory
from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.advanced import AdvancedStrategies
from ..backtester.vectorbt_engine import VectorBTEngine

logger = logging.getLogger(__name__)


class StrategySelector:
    """Select and manage active trading strategies."""
    
    def __init__(
        self,
        factory: StrategyFactory = None,
        sharpe_6m_threshold: float = 0.9
    ):
        """
        Initialize strategy selector.
        
        Args:
            factory: Strategy factory instance
            sharpe_6m_threshold: Minimum 6-month Sharpe for activation
        """
        self.factory = factory or StrategyFactory()
        self.sharpe_threshold = sharpe_6m_threshold
        self.active_strategies: List[Dict[str, Any]] = []
    
    async def refresh_active_strategies(
        self,
        prices: pd.DataFrame,
        max_correlation: float = 0.7
    ) -> int:
        """
        Refresh list of active strategies based on performance and correlation.
        
        Args:
            prices: Price DataFrame for return calculation
            max_correlation: Maximum correlation allowed between strategies
            
        Returns:
            Number of active strategies
        """
        # 1. Activate/deactivate based on current performance
        self.factory.activate_qualified_strategies(self.sharpe_threshold)
        
        # 2. Get all potential active strategies (unfiltered)
        potential_strategies = self.factory.get_active_strategies()
        
        if not potential_strategies:
            self.active_strategies = []
            return 0
            
        if max_correlation >= 1.0:
            self.active_strategies = potential_strategies
            logger.info(f"🎯 {len(self.active_strategies)} active strategies loaded (no corr filter)")
            return len(self.active_strategies)

        # 3. Filter by correlation
        logger.info(f"🔍 Filtering {len(potential_strategies)} potential strategies by correlation (threshold: {max_correlation})...")
        self.active_strategies = self._filter_by_correlation(potential_strategies, prices, max_correlation)
        
        logger.info(f"🎯 {len(self.active_strategies)} active strategies loaded after correlation filtering")
        return len(self.active_strategies)

    def _filter_by_correlation(
        self,
        strategies: List[Dict[str, Any]],
        prices: pd.DataFrame,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Greedily filter strategies by correlation.
        """
        if not strategies:
            return []
            
        # Sort by 6m Sharpe Ratio (Descending to prioritize top winners for fading)
        sorted_strategies = sorted(
            strategies,
            key=lambda x: x['performance']['sharpe_6m'],
            reverse=True
        )
        
        backtester = VectorBTEngine(prices)
        selected_strategies = []
        selected_returns = []
        
        for strategy in sorted_strategies:
            # Run a quick backtest to get returns
            result = backtester.run_backtest(strategy)
            rets = result.returns
            
            if rets is None or rets.empty or rets.std() == 0:
                continue
                
            # Check correlation with already selected strategies
            is_redundant = False
            for other_rets in selected_returns:
                corr = rets.corr(other_rets)
                if abs(corr) >= threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_strategies.append(strategy)
                selected_returns.append(rets)
                
        return selected_strategies
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        target_date: str = None,
        max_correlation: float = 0.7
    ) -> pd.DataFrame:
        """
        Generate signals for all active strategies.
        
        Args:
            prices: DataFrame with price data
            target_date: Date to generate signals for (default: latest)
            max_correlation: Max correlation for strategy selection
            
        Returns:
            DataFrame with signals per strategy
        """
        if not self.active_strategies:
            # We need to run the refresh synchronously here if possible, 
            # or ensure it's called before. Since we changed it to async, 
            # we should be careful. Actually, let's keep it simple for now.
            # I'll make refresh_active_strategies synchronous to avoid complexity.
            # (Changed from prospective 'async' to synchronous)
            self._refresh_sync(prices, max_correlation)
        
        if not self.active_strategies:
            logger.warning("⚠️ No active strategies")
            return pd.DataFrame()
        
        signals = []
        
        for strategy in self.active_strategies:
            try:
                signal = self._generate_strategy_signal(prices, strategy, target_date)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Signal generation failed for {strategy['strategy_id']}: {e}")
        
        if not signals:
            return pd.DataFrame()
        
        return pd.DataFrame(signals)
    
    def _generate_strategy_signal(
        self,
        prices: pd.DataFrame,
        strategy: Dict[str, Any],
        target_date: str = None
    ) -> Dict[str, Any]:
        """
        Generate signal for a single strategy.
        
        Args:
            prices: Price DataFrame
            strategy: Strategy configuration
            target_date: Target date
            
        Returns:
            Signal dict
        """
        asset = strategy['asset']
        strategy_type = strategy['strategy_type']
        strategy_name = strategy['strategy_name']
        params = strategy['params']
        related_asset = strategy.get('related_asset')
        
        # Generate entry/exit signals
        if strategy_type == 'momentum':
            entries, exits = MomentumStrategy.generate_signals(
                prices, asset, strategy_name, params, related_asset
            )
            # Momentum is typically long-only
            position = self._calculate_position_momentum(entries, exits)
        elif strategy_type == 'mean_reversion':
            long_entries, long_exits, short_entries, short_exits = \
                MeanReversionStrategy.generate_signals(
                    prices, asset, strategy_name, params, related_asset
                )
            position = self._calculate_position_mean_reversion(
                long_entries, long_exits, short_entries, short_exits
            )
        elif strategy_type == 'advanced':
            entries, exits = AdvancedStrategies.generate_signals(
                prices, asset, strategy_name, params, related_asset
            )
            # Treat advanced as long-only for now (consistent with vectorbt_engine update)
            position = self._calculate_position_momentum(entries, exits)
        else:
            raise ValueError(f"Unknown strategy category: {strategy_type}")
        
        # Get signal for target date
        if target_date:
            target_idx = pd.to_datetime(target_date)
            if target_idx in position.index:
                current_position = position.loc[target_idx]
            else:
                current_position = position.iloc[-1]
        else:
            current_position = position.iloc[-1]
        
        # FADE THE WINNER: Reverse signal ONLY for Rates. Keep original for FX.
        if 'Curncy' in asset:
            final_position = current_position  # FX remains original
        else:
            final_position = current_position * -1  # Rates are reversed
            
        return {
            'strategy_id': strategy['strategy_id'],
            'asset': asset,
            'related_asset': related_asset,
            'strategy_type': strategy_type,
            'strategy_name': strategy_name,
            'position': final_position,
            'sharpe_6m': strategy['performance']['sharpe_6m'],
            'date': target_date or str(position.index[-1].date()),
        }
    
    def _calculate_position_momentum(
        self,
        entries: pd.Series,
        exits: pd.Series
    ) -> pd.Series:
        """Calculate position series for momentum strategy."""
        position = pd.Series(0, index=entries.index)
        current = 0
        
        for i in range(len(entries)):
            if entries.iloc[i] and current == 0:
                current = 1
            elif exits.iloc[i] and current == 1:
                current = 0
            position.iloc[i] = current
        
        return position
    
    def _calculate_position_mean_reversion(
        self,
        long_entries: pd.Series,
        long_exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series
    ) -> pd.Series:
        """Calculate position series for mean reversion strategy."""
        position = pd.Series(0, index=long_entries.index)
        current = 0
        
        for i in range(len(long_entries)):
            if long_entries.iloc[i] and current == 0:
                current = 1
            elif short_entries.iloc[i] and current == 0:
                current = -1
            elif long_exits.iloc[i] and current == 1:
                current = 0
            elif short_exits.iloc[i] and current == -1:
                current = 0
            position.iloc[i] = current
        
        return position
    
    def aggregate_positions(
        self,
        signals: pd.DataFrame,
        method: str = 'sharpe_weighted'
    ) -> pd.DataFrame:
        """
        Aggregate signals into asset-level positions.
        
        Args:
            signals: DataFrame with strategy signals
            method: Aggregation method ('equal', 'sharpe_weighted', 'vote')
            
        Returns:
            DataFrame with asset positions
        """
        if signals.empty:
            return pd.DataFrame()
        
        # Group by asset
        asset_positions = []
        
        for asset in signals['asset'].unique():
            asset_signals = signals[signals['asset'] == asset]
            
            if method == 'equal':
                # Simple average
                avg_position = asset_signals['position'].mean()
            
            elif method == 'sharpe_weighted':
                # Sharpe-weighted average
                weights = asset_signals['sharpe_6m']
                if weights.sum() > 0:
                    avg_position = np.average(
                        asset_signals['position'],
                        weights=weights
                    )
                else:
                    avg_position = asset_signals['position'].mean()
            
            elif method == 'vote':
                # Majority vote
                positions = asset_signals['position'].values
                if np.sum(positions > 0) > np.sum(positions < 0):
                    avg_position = 1
                elif np.sum(positions < 0) > np.sum(positions > 0):
                    avg_position = -1
                else:
                    avg_position = 0
            
            else:
                avg_position = asset_signals['position'].mean()
            
            # Discretize to -1, 0, 1
            if avg_position > 0.3:
                final_position = 1
            elif avg_position < -0.3:
                final_position = -1
            else:
                final_position = 0
            
            asset_positions.append({
                'asset': asset,
                'position': final_position,
                'raw_position': avg_position,
                'num_strategies': len(asset_signals),
                'momentum_count': (asset_signals['strategy_type'] == 'momentum').sum(),
                'mr_count': (asset_signals['strategy_type'] == 'mean_reversion').sum(),
                'adv_count': (asset_signals['strategy_type'] == 'advanced').sum(),
                'avg_sharpe_6m': asset_signals['sharpe_6m'].mean(),
            })
        
        return pd.DataFrame(asset_positions)
    
    def get_trading_report(
        self,
        prices: pd.DataFrame,
        target_date: str = None,
        max_correlation: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading report.
        
        Args:
            prices: Price DataFrame
            target_date: Target date
            max_correlation: Max correlation for strategy selection
            
        Returns:
            Trading report dict
        """
        signals = self.generate_signals(prices, target_date, max_correlation)
        positions = self.aggregate_positions(signals)
        
        # Create summary including ALL assets in the price data
        asset_summary = []
        # Custom sort for assets: US, DE, UK, AU, JP, KR rates, then FX
        def asset_sort_key(ticker: str) -> tuple:
            # Group priority
            if 'USGG' in ticker: group = 1
            elif 'GDBR' in ticker: group = 2
            elif 'GUKG' in ticker: group = 3
            elif 'GTAUD' in ticker: group = 4
            elif 'GJGB' in ticker: group = 5
            elif 'GVSK' in ticker: group = 6
            elif 'Index' in ticker or 'Corp' in ticker: group = 7  # Other rates (FR, IT etc)
            elif 'Curncy' in ticker: group = 8  # FX
            else: group = 9
            
            return (group, ticker)

        all_assets = sorted(list(prices.columns), key=asset_sort_key)
        
        # Map existing positions for quick lookup
        pos_lookup = {row['asset']: row for _, row in positions.iterrows()} if not positions.empty else {}
        
        for asset in all_assets:
            if asset in pos_lookup:
                row = pos_lookup[asset]
                pos_str = '🟢 LONG' if row['position'] == 1 else \
                          '🔴 SHORT' if row['position'] == -1 else '⚪ FLAT'
                asset_summary.append({
                    'asset': asset,
                    'position': pos_str,
                    'confidence': abs(row['raw_position']),
                    'strategies': row['num_strategies'],
                    'momentum': row['momentum_count'],
                    'mean_reversion': row['mr_count'],
                    'advanced': row['adv_count'],
                })
            else:
                # Default "NOSIG" for assets with no active strategies
                asset_summary.append({
                    'asset': asset,
                    'position': '🔘 NOSIG',
                    'confidence': 0.0,
                    'strategies': 0,
                    'momentum': 0,
                    'mean_reversion': 0,
                    'advanced': 0,
                })
        
        return {
            'date': target_date or str(datetime.now().date()),
            'total_active_strategies': len(self.active_strategies),
            'signals_generated': len(signals),
            'asset_positions': asset_summary,
            'raw_signals': signals.to_dict('records') if not signals.empty else [],
            'aggregated_positions': positions.to_dict('records') if not positions.empty else [],
        }
    
    def get_position_for_asset(
        self,
        prices: pd.DataFrame,
        asset: str,
        target_date: str = None
    ) -> Tuple[int, float]:
        """
        Get aggregated position for a specific asset.
        
        Args:
            prices: Price DataFrame
            asset: Asset ticker
            target_date: Target date
            
        Returns:
            Tuple of (position, confidence)
        """
        signals = self.generate_signals(prices, target_date)
        
        if signals.empty:
            return 0, 0.0
        
        asset_signals = signals[signals['asset'] == asset]
        
        if asset_signals.empty:
            return 0, 0.0
        
        positions = self.aggregate_positions(asset_signals)
        
        if positions.empty:
            return 0, 0.0
        
        row = positions.iloc[0]
        return int(row['position']), abs(row['raw_position'])
    
    def _refresh_sync(self, prices: pd.DataFrame, max_correlation: float = 0.7):
        """Internal synchronous refresh."""
        # This is just a helper to keep the API clean if we don't want async everywhere
        self.factory.activate_qualified_strategies(self.sharpe_threshold)
        potential_strategies = self.factory.get_active_strategies()
        
        if not potential_strategies:
            self.active_strategies = []
            return
            
        if max_correlation >= 1.0:
            self.active_strategies = potential_strategies
            return

        self.active_strategies = self._filter_by_correlation(potential_strategies, prices, max_correlation)
