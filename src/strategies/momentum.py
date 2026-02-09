"""
Momentum Strategy Logic for Global Macro Trading

Implements momentum-based trading signals using technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..indicators.technical import TechnicalIndicators
from ..indicators.cross_asset import CrossAssetIndicators


class MomentumStrategy:
    """Generate momentum-based trading signals."""
    
    STRATEGY_TYPES = [
        'ma_crossover',
        'rsi_momentum',
        'macd_crossover',
        'breakout',
        'rate_of_change',
        'spread_momentum',
    ]
    
    @staticmethod
    def ma_crossover(
        prices: pd.Series,
        fast_period: int = 10,
        slow_period: int = 30
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Moving average crossover strategy.
        
        Args:
            prices: Price series
            fast_period: Fast MA period
            slow_period: Slow MA period
            
        Returns:
            Tuple of (entries, exits) as boolean series
        """
        fast_ma = TechnicalIndicators.sma(prices, fast_period)
        slow_ma = TechnicalIndicators.sma(prices, slow_period)
        
        # Long entry: fast crosses above slow
        # Long exit: fast crosses below slow
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        return entries, exits
    
    @staticmethod
    def ema_crossover(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26
    ) -> Tuple[pd.Series, pd.Series]:
        """
        EMA crossover strategy.
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            Tuple of (entries, exits)
        """
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)
        
        entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
        exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
        
        return entries, exits
    
    @staticmethod
    def rsi_momentum(
        prices: pd.Series,
        period: int = 14,
        entry_threshold: float = 50,
        exit_threshold: float = 50
    ) -> Tuple[pd.Series, pd.Series]:
        """
        RSI momentum strategy (RSI > threshold = bullish momentum).
        
        Args:
            prices: Price series
            period: RSI period
            entry_threshold: RSI level for entry
            exit_threshold: RSI level for exit
            
        Returns:
            Tuple of (entries, exits)
        """
        rsi = TechnicalIndicators.rsi(prices, period)
        
        # Entry when RSI crosses above threshold
        entries = (rsi > entry_threshold) & (rsi.shift(1) <= entry_threshold)
        # Exit when RSI crosses below threshold
        exits = (rsi < exit_threshold) & (rsi.shift(1) >= exit_threshold)
        
        return entries, exits
    
    @staticmethod
    def macd_crossover(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """
        MACD crossover strategy.
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (entries, exits)
        """
        macd_line, signal_line, _ = TechnicalIndicators.macd(
            prices, fast_period, slow_period, signal_period
        )
        
        # Entry: MACD crosses above signal
        entries = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        # Exit: MACD crosses below signal
        exits = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        return entries, exits
    
    @staticmethod
    def breakout(
        prices: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Donchian channel breakout strategy.
        
        Args:
            prices: Price series
            period: Channel period
            
        Returns:
            Tuple of (entries, exits)
        """
        upper, middle, lower = TechnicalIndicators.donchian_channel(prices, period)
        
        # Entry: price breaks above upper band
        entries = (prices > upper.shift(1))
        # Exit: price breaks below middle or lower
        exits = (prices < middle.shift(1))
        
        return entries, exits
    
    @staticmethod
    def rate_of_change(
        prices: pd.Series,
        period: int = 20,
        entry_threshold: float = 0,
        exit_threshold: float = 0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Rate of change momentum strategy.
        
        Args:
            prices: Price series
            period: ROC period
            entry_threshold: ROC threshold for entry
            exit_threshold: ROC threshold for exit
            
        Returns:
            Tuple of (entries, exits)
        """
        roc = TechnicalIndicators.rate_of_change(prices, period)
        
        entries = (roc > entry_threshold) & (roc.shift(1) <= entry_threshold)
        exits = (roc < exit_threshold) & (roc.shift(1) >= exit_threshold)
        
        return entries, exits
    
    @staticmethod
    def spread_momentum(
        prices: pd.DataFrame,
        asset1: str,
        asset2: str,
        period: int = 20,
        threshold: float = 0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Spread momentum strategy.
        
        Args:
            prices: DataFrame with price data
            asset1: First asset ticker
            asset2: Second asset ticker
            period: Momentum period
            threshold: Entry/exit threshold
            
        Returns:
            Tuple of (entries, exits)
        """
        momentum = CrossAssetIndicators.spread_momentum(
            prices[asset1], prices[asset2], period
        )
        
        entries = (momentum > threshold) & (momentum.shift(1) <= threshold)
        exits = (momentum < -threshold) & (momentum.shift(1) >= -threshold)
        
        return entries, exits
    
    @classmethod
    def generate_signals(
        cls,
        prices: pd.DataFrame,
        asset: str,
        strategy_type: str,
        params: Dict[str, Any],
        related_asset: Optional[str] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate momentum signals based on strategy type.
        
        Args:
            prices: DataFrame with price data
            asset: Target asset ticker
            strategy_type: Type of momentum strategy
            params: Strategy parameters
            related_asset: Related asset for cross-asset strategies
            
        Returns:
            Tuple of (entries, exits)
        """
        price_series = prices[asset]
        
        if strategy_type == 'ma_crossover':
            return cls.ma_crossover(price_series, **params)
        elif strategy_type == 'ema_crossover':
            return cls.ema_crossover(price_series, **params)
        elif strategy_type == 'rsi_momentum':
            return cls.rsi_momentum(price_series, **params)
        elif strategy_type == 'macd_crossover':
            return cls.macd_crossover(price_series, **params)
        elif strategy_type == 'breakout':
            return cls.breakout(price_series, **params)
        elif strategy_type == 'rate_of_change':
            return cls.rate_of_change(price_series, **params)
        elif strategy_type == 'spread_momentum' and related_asset:
            return cls.spread_momentum(prices, asset, related_asset, **params)
        else:
            raise ValueError(f"Unknown momentum strategy: {strategy_type}")
