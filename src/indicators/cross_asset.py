"""
Cross-Asset Indicators for Global Macro Trading

Calculates spread, ratio, and correlation-based indicators for cross-asset analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class CrossAssetIndicators:
    """Calculate cross-asset indicators for macro trading strategies."""
    
    @staticmethod
    def spread(asset1: pd.Series, asset2: pd.Series) -> pd.Series:
        """
        Calculate simple spread between two assets.
        
        Args:
            asset1: First asset prices/yields
            asset2: Second asset prices/yields
            
        Returns:
            Spread series (asset1 - asset2)
        """
        return asset1 - asset2
    
    @staticmethod
    def spread_zscore(
        asset1: pd.Series,
        asset2: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Z-Score of spread between two assets.
        
        Args:
            asset1: First asset prices/yields
            asset2: Second asset prices/yields
            period: Lookback period for Z-Score
            
        Returns:
            Spread Z-Score series
        """
        spread = CrossAssetIndicators.spread(asset1, asset2)
        mean = spread.rolling(window=period).mean()
        std = spread.rolling(window=period).std()
        
        return (spread - mean) / std
    
    @staticmethod
    def spread_percentile(
        asset1: pd.Series,
        asset2: pd.Series,
        period: int = 252
    ) -> pd.Series:
        """
        Calculate percentile rank of spread. Faster implementation using numpy.
        """
        spread = CrossAssetIndicators.spread(asset1, asset2)
        
        def pct_rank_fast(x):
            if np.isnan(x).any():
                return np.nan
            # Count how many elements are smaller than the last one
            last_val = x[-1]
            smaller = np.sum(x < last_val)
            equal = np.sum(x == last_val)
            # Standard rank: smaller + (equal + 1) / 2
            rank = smaller + (equal + 1) / 2.0
            return (rank / len(x)) * 100
        
        return spread.rolling(window=period).apply(pct_rank_fast, raw=True)
    
    @staticmethod
    def ratio(asset1: pd.Series, asset2: pd.Series) -> pd.Series:
        """
        Calculate ratio between two assets.
        
        Args:
            asset1: First asset prices
            asset2: Second asset prices
            
        Returns:
            Ratio series (asset1 / asset2)
        """
        return asset1 / asset2
    
    @staticmethod
    def ratio_zscore(
        asset1: pd.Series,
        asset2: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Z-Score of ratio between two assets.
        
        Args:
            asset1: First asset prices
            asset2: Second asset prices
            period: Lookback period
            
        Returns:
            Ratio Z-Score series
        """
        ratio = CrossAssetIndicators.ratio(asset1, asset2)
        mean = ratio.rolling(window=period).mean()
        std = ratio.rolling(window=period).std()
        
        return (ratio - mean) / std
    
    @staticmethod
    def log_ratio(asset1: pd.Series, asset2: pd.Series) -> pd.Series:
        """
        Calculate log ratio between two assets.
        
        Args:
            asset1: First asset prices
            asset2: Second asset prices
            
        Returns:
            Log ratio series
        """
        return np.log(asset1 / asset2)
    
    @staticmethod
    def correlation_rolling(
        asset1: pd.Series,
        asset2: pd.Series,
        period: int = 60
    ) -> pd.Series:
        """
        Calculate rolling correlation between two assets.
        
        Args:
            asset1: First asset prices/returns
            asset2: Second asset prices/returns
            period: Rolling window
            
        Returns:
            Rolling correlation series
        """
        # Use returns for correlation
        returns1 = asset1.pct_change()
        returns2 = asset2.pct_change()
        
        return returns1.rolling(window=period).corr(returns2)
    
    @staticmethod
    def relative_strength(
        asset1: pd.Series,
        asset2: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate relative strength of asset1 vs asset2.
        
        Args:
            asset1: First asset prices
            asset2: Second asset prices
            period: Lookback period
            
        Returns:
            Relative strength series
        """
        returns1 = asset1.pct_change(period)
        returns2 = asset2.pct_change(period)
        
        return returns1 - returns2
    
    @staticmethod
    def beta_rolling(
        asset: pd.Series,
        benchmark: pd.Series,
        period: int = 60
    ) -> pd.Series:
        """
        Calculate rolling beta of asset vs benchmark.
        
        Args:
            asset: Asset prices
            benchmark: Benchmark prices
            period: Rolling window
            
        Returns:
            Rolling beta series
        """
        returns_asset = asset.pct_change()
        returns_bench = benchmark.pct_change()
        
        covariance = returns_asset.rolling(window=period).cov(returns_bench)
        variance = returns_bench.rolling(window=period).var()
        
        return covariance / variance
    
    @staticmethod
    def cross_sectional_rank(prices: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate cross-sectional rank of assets based on returns.
        
        Args:
            prices: DataFrame with multiple asset prices
            period: Return lookback period
            
        Returns:
            DataFrame where each value is the rank of that asset at that time
        """
        returns = prices.pct_change(period)
        return returns.rank(axis=1, ascending=False)
    
    @staticmethod
    def spread_momentum(
        asset1: pd.Series,
        asset2: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate momentum of spread.
        
        Args:
            asset1: First asset prices/yields
            asset2: Second asset prices/yields
            period: Momentum period
            
        Returns:
            Spread momentum series
        """
        spread = CrossAssetIndicators.spread(asset1, asset2)
        return spread - spread.shift(period)
    
    @staticmethod
    def spread_rsi(
        asset1: pd.Series,
        asset2: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate RSI of spread.
        
        Args:
            asset1: First asset prices/yields
            asset2: Second asset prices/yields
            period: RSI period
            
        Returns:
            Spread RSI series
        """
        spread = CrossAssetIndicators.spread(asset1, asset2)
        
        delta = spread.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @classmethod
    def calculate(
        cls,
        prices: pd.DataFrame,
        indicator_name: str,
        asset1_ticker: str,
        asset2_ticker: str,
        params: Dict[str, Any]
    ) -> pd.Series:
        """
        Calculate cross-asset indicator by name.
        
        Args:
            prices: DataFrame with all price data
            indicator_name: Name of indicator
            asset1_ticker: First asset ticker
            asset2_ticker: Second asset ticker
            params: Indicator parameters
            
        Returns:
            Indicator series
        """
        asset1 = prices[asset1_ticker]
        asset2 = prices[asset2_ticker]
        
        indicator_map = {
            'spread': cls.spread,
            'spread_zscore': cls.spread_zscore,
            'spread_percentile': cls.spread_percentile,
            'ratio': cls.ratio,
            'ratio_zscore': cls.ratio_zscore,
            'log_ratio': cls.log_ratio,
            'correlation_rolling': cls.correlation_rolling,
            'relative_strength': cls.relative_strength,
            'beta_rolling': cls.beta_rolling,
            'spread_momentum': cls.spread_momentum,
            'spread_rsi': cls.spread_rsi,
        }
        
        if indicator_name not in indicator_map:
            raise ValueError(f"Unknown cross-asset indicator: {indicator_name}")
        
        func = indicator_map[indicator_name]
        
        # Simple indicators without params
        if indicator_name in ['spread', 'ratio', 'log_ratio']:
            return func(asset1, asset2)
        
        return func(asset1, asset2, **params)
