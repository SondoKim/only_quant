"""
Mean Reversion Strategy Logic for Global Macro Trading

Implements mean-reversion trading signals using technical and cross-asset indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..indicators.technical import TechnicalIndicators
from ..indicators.cross_asset import CrossAssetIndicators


class MeanReversionStrategy:
    """Generate mean-reversion trading signals."""
    
    STRATEGY_TYPES = [
        'zscore_reversion',
        'rsi_extremes',
        'bollinger_reversion',
        'spread_zscore_reversion',
        'spread_percentile_reversion',
        'spread_rsi_reversion',
    ]
    
    @staticmethod
    def zscore_reversion(
        prices: pd.Series,
        period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Z-Score mean reversion strategy.
        
        Args:
            prices: Price series
            period: Z-Score period
            entry_threshold: Z-Score for entry
            exit_threshold: Z-Score for exit
            
        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        zscore = TechnicalIndicators.zscore(prices, period)
        
        # Long: Z-Score < -threshold (oversold)
        long_entries = (zscore < -entry_threshold) & (zscore.shift(1) >= -entry_threshold)
        long_exits = (zscore > -exit_threshold) & (zscore.shift(1) <= -exit_threshold)
        
        # Short: Z-Score > threshold (overbought)
        short_entries = (zscore > entry_threshold) & (zscore.shift(1) <= entry_threshold)
        short_exits = (zscore < exit_threshold) & (zscore.shift(1) >= exit_threshold)
        
        return long_entries, long_exits, short_entries, short_exits
    
    @staticmethod
    def rsi_extremes(
        prices: pd.Series,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        exit_level: float = 50
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        RSI extremes mean reversion strategy.
        
        Args:
            prices: Price series
            period: RSI period
            oversold: Oversold level
            overbought: Overbought level
            exit_level: Exit level (typically 50)
            
        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        rsi = TechnicalIndicators.rsi(prices, period)
        
        # Long: RSI crosses up from oversold
        long_entries = (rsi > oversold) & (rsi.shift(1) <= oversold)
        long_exits = (rsi > exit_level) & (rsi.shift(1) <= exit_level)
        
        # Short: RSI crosses down from overbought
        short_entries = (rsi < overbought) & (rsi.shift(1) >= overbought)
        short_exits = (rsi < exit_level) & (rsi.shift(1) >= exit_level)
        
        return long_entries, long_exits, short_entries, short_exits
    
    @staticmethod
    def bollinger_reversion(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Band mean reversion strategy.
        
        Args:
            prices: Price series
            period: Bollinger period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, period, std_dev)
        
        # Long: price touches lower band
        long_entries = (prices <= lower) & (prices.shift(1) > lower.shift(1))
        long_exits = (prices >= middle) & (prices.shift(1) < middle.shift(1))
        
        # Short: price touches upper band
        short_entries = (prices >= upper) & (prices.shift(1) < upper.shift(1))
        short_exits = (prices <= middle) & (prices.shift(1) > middle.shift(1))
        
        return long_entries, long_exits, short_entries, short_exits
    
    @staticmethod
    def spread_zscore_reversion(
        prices: pd.DataFrame,
        asset1: str,
        asset2: str,
        period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Spread Z-Score mean reversion strategy.
        
        Args:
            prices: DataFrame with price data
            asset1: First asset ticker
            asset2: Second asset ticker
            period: Z-Score period
            entry_threshold: Entry threshold
            exit_threshold: Exit threshold
            
        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        zscore = CrossAssetIndicators.spread_zscore(
            prices[asset1], prices[asset2], period
        )
        
        # Long asset1/short asset2: spread Z-Score < -threshold
        long_entries = (zscore < -entry_threshold) & (zscore.shift(1) >= -entry_threshold)
        long_exits = (zscore > -exit_threshold) & (zscore.shift(1) <= -exit_threshold)
        
        # Short asset1/long asset2: spread Z-Score > threshold
        short_entries = (zscore > entry_threshold) & (zscore.shift(1) <= entry_threshold)
        short_exits = (zscore < exit_threshold) & (zscore.shift(1) >= exit_threshold)
        
        return long_entries, long_exits, short_entries, short_exits
    
    @staticmethod
    def spread_percentile_reversion(
        prices: pd.DataFrame,
        asset1: str,
        asset2: str,
        period: int = 252,
        low_percentile: float = 20,
        high_percentile: float = 80,
        exit_percentile: float = 50
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Spread percentile mean reversion strategy.
        
        Args:
            prices: DataFrame with price data
            asset1: First asset ticker
            asset2: Second asset ticker
            period: Lookback period
            low_percentile: Low percentile for long entry
            high_percentile: High percentile for short entry
            exit_percentile: Exit percentile
            
        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        pct = CrossAssetIndicators.spread_percentile(
            prices[asset1], prices[asset2], period
        )
        
        # Long: percentile below low threshold
        long_entries = (pct < low_percentile) & (pct.shift(1) >= low_percentile)
        long_exits = (pct > exit_percentile) & (pct.shift(1) <= exit_percentile)
        
        # Short: percentile above high threshold
        short_entries = (pct > high_percentile) & (pct.shift(1) <= high_percentile)
        short_exits = (pct < exit_percentile) & (pct.shift(1) >= exit_percentile)
        
        return long_entries, long_exits, short_entries, short_exits
    
    @staticmethod
    def spread_rsi_reversion(
        prices: pd.DataFrame,
        asset1: str,
        asset2: str,
        period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        exit_level: float = 50
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Spread RSI mean reversion strategy.
        
        Args:
            prices: DataFrame with price data
            asset1: First asset ticker
            asset2: Second asset ticker
            period: RSI period
            oversold: Oversold level
            overbought: Overbought level
            exit_level: Exit level
            
        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        rsi = CrossAssetIndicators.spread_rsi(
            prices[asset1], prices[asset2], period
        )
        
        # Long: RSI crosses up from oversold
        long_entries = (rsi > oversold) & (rsi.shift(1) <= oversold)
        long_exits = (rsi > exit_level) & (rsi.shift(1) <= exit_level)
        
        # Short: RSI crosses down from overbought
        short_entries = (rsi < overbought) & (rsi.shift(1) >= overbought)
        short_exits = (rsi < exit_level) & (rsi.shift(1) >= exit_level)
        
        return long_entries, long_exits, short_entries, short_exits
    
    @classmethod
    def generate_signals(
        cls,
        prices: pd.DataFrame,
        asset: str,
        strategy_type: str,
        params: Dict[str, Any],
        related_asset: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Generate mean reversion signals.
        
        Args:
            prices: DataFrame with price data
            asset: Target asset ticker
            strategy_type: Type of mean reversion strategy
            params: Strategy parameters
            related_asset: Related asset for spread strategies
            
        Returns:
            Tuple of (long_entries, long_exits, short_entries, short_exits)
        """
        price_series = prices[asset]
        
        if strategy_type == 'zscore_reversion':
            return cls.zscore_reversion(price_series, **params)
        elif strategy_type == 'rsi_extremes':
            return cls.rsi_extremes(price_series, **params)
        elif strategy_type == 'bollinger_reversion':
            return cls.bollinger_reversion(price_series, **params)
        elif strategy_type == 'spread_zscore_reversion' and related_asset:
            return cls.spread_zscore_reversion(prices, asset, related_asset, **params)
        elif strategy_type == 'spread_percentile_reversion' and related_asset:
            return cls.spread_percentile_reversion(prices, asset, related_asset, **params)
        elif strategy_type == 'spread_rsi_reversion' and related_asset:
            return cls.spread_rsi_reversion(prices, asset, related_asset, **params)
        else:
            raise ValueError(f"Unknown mean reversion strategy: {strategy_type}")
