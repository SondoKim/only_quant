"""
Advanced Strategy Logic for Global Macro Trading

Implements 5 advanced, fast-testable strategy types.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from ..indicators.technical import TechnicalIndicators
from ..indicators.cross_asset import CrossAssetIndicators


class AdvancedStrategies:
    """Generate advanced trading signals."""
    
    STRATEGY_TYPES = [
        'filtered_momentum',
        'lead_lag_momentum',
        'multi_tf_momentum',
        'volatility_breakout',
        'relative_strength_rank',
    ]
    
    @staticmethod
    def filtered_momentum(
        prices: pd.Series,
        ma_period: int = 200,
        rsi_period: int = 14,
        rsi_entry: float = 50,
        rsi_exit: float = 50
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Trend-filtered momentum: Buy RSI > entry only if price > MA.
        """
        ma = TechnicalIndicators.sma(prices, ma_period)
        rsi = TechnicalIndicators.rsi(prices, rsi_period)
        
        trend_ok = prices > ma
        
        entries = trend_ok & (rsi > rsi_entry) & (rsi.shift(1) <= rsi_entry)
        exits = (rsi < rsi_exit) & (rsi.shift(1) >= rsi_exit)
        
        return entries, exits

    @staticmethod
    def lead_lag_momentum(
        prices: pd.DataFrame,
        target_asset: str,
        lead_asset: str,
        period: int = 20,
        threshold: float = 0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Trade target asset based on lead asset momentum.
        """
        lead_prices = prices[lead_asset]
        lead_roc = TechnicalIndicators.rate_of_change(lead_prices, period)
        
        entries = (lead_roc > threshold) & (lead_roc.shift(1) <= threshold)
        exits = (lead_roc < -threshold) & (lead_roc.shift(1) >= -threshold)
        
        return entries, exits

    @staticmethod
    def multi_tf_momentum(
        prices: pd.Series,
        periods: List[int] = [20, 60, 120]
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Aggregate momentum across multiple timeframes.
        """
        moms = []
        for p in periods:
            moms.append(TechnicalIndicators.rate_of_change(prices, p))
            
        combined_mom = pd.concat(moms, axis=1).mean(axis=1)
        
        entries = (combined_mom > 0) & (combined_mom.shift(1) <= 0)
        exits = (combined_mom < 0) & (combined_mom.shift(1) >= 0)
        
        return entries, exits

    @staticmethod
    def volatility_breakout(
        prices: pd.DataFrame,
        asset: str,
        period: int = 20,
        k: float = 1.0
    ) -> Tuple[pd.Series, pd.Series]:
        """
        ATR-based volatility breakout from previous close.
        """
        # Note: In our current data loader, we might only have Close.
        # If High/Low aren't available, we use StdDev as a fallback for ATR.
        close = prices[asset]
        
        # Fallback to StdDev if High/Low not in columns
        if 'High' in prices.columns and 'Low' in prices.columns:
            high = prices['High']
            low = prices['Low']
            atr = TechnicalIndicators.atr(high.get(asset, high), low.get(asset, low), close, period)
        else:
            atr = close.rolling(period).std()
            
        upper = close.shift(1) + (atr.shift(1) * k)
        lower = close.shift(1) - (atr.shift(1) * k)
        
        entries = (close > upper) & (close.shift(1) <= upper.shift(1))
        exits = (close < lower) & (close.shift(1) >= lower.shift(1))
        
        return entries, exits

    @staticmethod
    def relative_strength_rank(
        prices: pd.DataFrame,
        asset: str,
        period: int = 20,
        top_n: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Buy if asset is in the top N performers of the basket.
        """
        ranks = CrossAssetIndicators.cross_sectional_rank(prices, period)
        asset_rank = ranks[asset]
        
        entries = (asset_rank <= top_n) & (asset_rank.shift(1) > top_n)
        exits = (asset_rank > top_n) & (asset_rank.shift(1) <= top_n)
        
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
        """Generate advanced signals."""
        price_series = prices[asset]
        
        if strategy_type == 'filtered_momentum':
            return cls.filtered_momentum(price_series, **params)
        elif strategy_type == 'lead_lag_momentum' and related_asset:
            return cls.lead_lag_momentum(prices, asset, related_asset, **params)
        elif strategy_type == 'multi_tf_momentum':
            return cls.multi_tf_momentum(price_series, **params)
        elif strategy_type == 'volatility_breakout':
            return cls.volatility_breakout(prices, asset, **params)
        elif strategy_type == 'relative_strength_rank':
            return cls.relative_strength_rank(prices, asset, **params)
        else:
            raise ValueError(f"Unknown advanced strategy: {strategy_type}")
