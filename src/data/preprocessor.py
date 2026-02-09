"""
Data Preprocessor for Global Macro Trading

Handles data cleaning, returns calculation, and normalization.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess price data for strategy development."""
    
    def __init__(self, prices: pd.DataFrame):
        """
        Initialize preprocessor with price data.
        
        Args:
            prices: DataFrame with price data indexed by date
        """
        self.prices = prices.copy()
        self.returns: Optional[pd.DataFrame] = None
        self.log_returns: Optional[pd.DataFrame] = None
        
    def clean(self, method: str = 'ffill') -> 'DataPreprocessor':
        """
        Clean data by handling missing values.
        
        Args:
            method: Method for filling missing values ('ffill', 'interpolate')
            
        Returns:
            self for method chaining
        """
        if method == 'ffill':
            self.prices = self.prices.ffill()
        elif method == 'interpolate':
            self.prices = self.prices.interpolate(method='time')
        
        # Drop any remaining NaN rows at the start
        self.prices = self.prices.dropna()
        
        logger.info(f"✅ Data cleaned: {self.prices.shape}")
        return self
    
    def calculate_returns(self, periods: int = 1) -> 'DataPreprocessor':
        """
        Calculate simple returns.
        
        Args:
            periods: Number of periods for return calculation
            
        Returns:
            self for method chaining
        """
        self.returns = self.prices.pct_change(periods=periods)
        return self
    
    def calculate_log_returns(self, periods: int = 1) -> 'DataPreprocessor':
        """
        Calculate log returns.
        
        Args:
            periods: Number of periods for return calculation
            
        Returns:
            self for method chaining
        """
        self.log_returns = np.log(self.prices / self.prices.shift(periods))
        return self
    
    def normalize(self, method: str = 'zscore', window: int = 252) -> pd.DataFrame:
        """
        Normalize price data.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'percentile')
            window: Rolling window for normalization
            
        Returns:
            Normalized DataFrame
        """
        if method == 'zscore':
            mean = self.prices.rolling(window=window).mean()
            std = self.prices.rolling(window=window).std()
            return (self.prices - mean) / std
        
        elif method == 'minmax':
            min_val = self.prices.rolling(window=window).min()
            max_val = self.prices.rolling(window=window).max()
            return (self.prices - min_val) / (max_val - min_val)
        
        elif method == 'percentile':
            def percentile_rank(x):
                return pd.Series(x).rank(pct=True).iloc[-1]
            return self.prices.rolling(window=window).apply(percentile_rank)
        
        return self.prices
    
    def split_train_test(
        self,
        train_end: str,
        test_start: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            train_end: End date for training data
            test_start: Start date for test data (default: day after train_end)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_end = pd.to_datetime(train_end)
        
        if test_start is None:
            test_start = train_end + pd.Timedelta(days=1)
        else:
            test_start = pd.to_datetime(test_start)
        
        train = self.prices[self.prices.index <= train_end]
        test = self.prices[self.prices.index >= test_start]
        
        return train, test
    
    def get_data(self) -> pd.DataFrame:
        """Get processed price data."""
        return self.prices
    
    def get_returns(self) -> pd.DataFrame:
        """Get returns data."""
        if self.returns is None:
            self.calculate_returns()
        return self.returns
    
    def get_log_returns(self) -> pd.DataFrame:
        """Get log returns data."""
        if self.log_returns is None:
            self.calculate_log_returns()
        return self.log_returns
