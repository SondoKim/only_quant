"""
Technical Indicators for Global Macro Trading

Provides vectorized technical indicator calculations optimized for fast backtesting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


class TechnicalIndicators:
    """Calculate technical indicators for trading strategies."""
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            SMA series
        """
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            EMA series
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            prices: Price series
            period: Lookback period (default: 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = prices.diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            prices: Price series
            period: Lookback period
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def bollinger_pct(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.Series:
        """
        Bollinger Band %B indicator.
        
        Args:
            prices: Price series
            period: Lookback period
            std_dev: Number of standard deviations
            
        Returns:
            %B series (0-1 typically, can exceed)
        """
        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, period, std_dev)
        return (prices - lower) / (upper - lower)
    
    @staticmethod
    def bollinger_width(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.Series:
        """
        Bollinger Band Width indicator (volatility).
        
        Args:
            prices: Price series
            period: Lookback period
            std_dev: Number of standard deviations
            
        Returns:
            Band width series
        """
        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, period, std_dev)
        return (upper - lower) / middle

    @staticmethod
    def rolling_rank(prices: pd.Series, period: int = 252) -> pd.Series:
        """
        Rolling rank of the current price compared to historical.
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Rolling rank series
        """
        return prices.rolling(window=period).apply(lambda x: pd.Series(x).rank().iloc[-1], raw=False)
    
    @staticmethod
    def zscore(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Rolling Z-Score.
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Z-Score series
        """
        mean = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return (prices - mean) / std
    
    @staticmethod
    def macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = TechnicalIndicators.ema(prices, fast_period)
        slow_ema = TechnicalIndicators.ema(prices, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rate_of_change(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change (ROC).
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            ROC series (percentage)
        """
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100
    
    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum indicator.
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Momentum series
        """
        return prices - prices.shift(period)
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def percentile_rank(prices: pd.Series, period: int = 252) -> pd.Series:
        """
        Rolling percentile rank.
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Percentile rank series (0-100)
        """
        def pct_rank(x):
            return pd.Series(x).rank(pct=True).iloc[-1] * 100
        
        return prices.rolling(window=period).apply(pct_rank, raw=False)
    
    @staticmethod
    def adx(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX) - measures trend strength.
        
        Uses price changes as proxy for High/Low when only Close data is available
        (common for bond yields and macro data).
        
        Args:
            prices: Price series (close)
            period: ADX period
            
        Returns:
            ADX series (0-100, >25 = trending)
        """
        # Use absolute price changes as proxy for directional movement
        diff = prices.diff()
        
        # Positive/Negative directional movement
        plus_dm = diff.where(diff > 0, 0.0)
        minus_dm = (-diff).where(diff < 0, 0.0)
        
        # True range proxy (absolute change)
        tr = diff.abs()
        
        # Wilder's smoothing
        alpha = 1 / period
        smoothed_tr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smoothed_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        smoothed_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        # DI+ and DI-
        plus_di = 100 * smoothed_plus / smoothed_tr.replace(0, np.nan)
        minus_di = 100 * smoothed_minus / smoothed_tr.replace(0, np.nan)
        
        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        
        return adx.fillna(0)
    
    @staticmethod
    def donchian_channel(
        prices: pd.Series,
        period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channel.
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Tuple of (upper, middle, lower)
        """
        upper = prices.rolling(window=period).max()
        lower = prices.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    @classmethod
    def calculate(
        cls,
        prices: pd.Series,
        indicator_name: str,
        params: Dict[str, Any]
    ) -> pd.Series:
        """
        Calculate indicator by name with parameters.
        
        Args:
            prices: Price series
            indicator_name: Name of indicator
            params: Indicator parameters
            
        Returns:
            Indicator series
        """
        indicator_map = {
            'sma': cls.sma,
            'ema': cls.ema,
            'rsi': cls.rsi,
            'zscore': cls.zscore,
            'rate_of_change': cls.rate_of_change,
            'momentum': cls.momentum,
            'percentile_rank': cls.percentile_rank,
        }
        
        if indicator_name in indicator_map:
            return indicator_map[indicator_name](prices, **params)
        
        # Handle special cases with multiple outputs
        if indicator_name == 'bollinger_pct':
            return cls.bollinger_pct(prices, **params)
        
        if indicator_name == 'macd':
            macd_line, _, _ = cls.macd(prices, **params)
            return macd_line
        
        if indicator_name == 'macd_signal':
            _, signal_line, _ = cls.macd(prices, **params)
            return signal_line
        
        if indicator_name == 'macd_hist':
            _, _, histogram = cls.macd(prices, **params)
            return histogram
        
        if indicator_name == 'rolling_sharpe':
            return cls.rolling_sharpe(prices, **params)
        
        if indicator_name == 'trailing_stop_signal':
            return cls.trailing_stop_signal(prices, **params)
        
        raise ValueError(f"Unknown indicator: {indicator_name}")

    @staticmethod
    def rolling_sharpe(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculates the annualized rolling Sharpe ratio.
        
        Args:
            prices: Price series (e.g., bond yields or cumulative returns)
            period: Lookback period
            
        Returns:
            Rolling Sharpe ratio series
        """
        # For rates, returns are often bps or daily diff
        returns = prices.diff().fillna(0)
        
        rolling_mean = returns.rolling(window=period).mean()
        rolling_std = returns.rolling(window=period).std()
        
        # Annualization factor (approx 252 trading days)
        sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        return sharpe.fillna(0)

    @staticmethod
    def trailing_stop_signal(series: pd.Series, stop_pct: float = 0.2, rolling_window: int = 60, abs_drop: float = 0.5) -> pd.Series:
        """
        Generates a signal when a series drops below its recent rolling high water mark.
        
        Args:
            series: Data series to track (e.g., Rolling Sharpe)
            stop_pct: Percentage drop from peak for signal (0.1 = 10%)
            rolling_window: Window for calculating the High Water Mark
            abs_drop: Minimum absolute drop to trigger (default: 0.5)
            
        Returns:
            Boolean series: True if stop breached
        """
        # Calculate Rolling High Water Mark over the last N days
        hwm = series.rolling(window=rolling_window, min_periods=1).max()
        
        breach_pct = (series < (hwm * (1 - stop_pct))) & (hwm > 0)
        breach_abs = (series < (hwm - abs_drop))
        
        return breach_pct | breach_abs
