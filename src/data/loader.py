"""
Bloomberg Data Loader for Global Macro Trading

Loads historical price data from Bloomberg using xbbg library.
Includes caching and offline fallback support.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bloomberg xbbg import
try:
    from xbbg import blp
    XBBG_AVAILABLE = True
except ImportError:
    XBBG_AVAILABLE = False
    logger.warning("xbbg not installed. Bloomberg data loading disabled.")


class DataLoader:
    """Load and cache Bloomberg data for global macro trading."""
    
    def __init__(self, config_path: str = None, cache_dir: str = None):
        """
        Initialize DataLoader.
        
        Args:
            config_path: Path to assets.yaml configuration
            cache_dir: Directory for caching data
        """
        self.config_path = config_path or self._get_default_config_path()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent.parent / 'data' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.all_tickers = self._extract_all_tickers()
        
    def _get_default_config_path(self) -> str:
        """Get default path to assets.yaml."""
        return str(Path(__file__).parent.parent.parent / 'config' / 'assets.yaml')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load asset configuration from YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
    
    def _extract_all_tickers(self) -> List[str]:
        """Extract all tickers from configuration."""
        tickers = []
        
        # Rates tickers
        if 'rates' in self.config:
            for country, data in self.config['rates'].items():
                if 'tickers' in data:
                    tickers.extend(data['tickers'])
        
        # FX tickers
        if 'fx' in self.config:
            for currency, data in self.config['fx'].items():
                if 'ticker' in data:
                    tickers.append(data['ticker'])
        
        return list(set(tickers))  # Remove duplicates
    
    def load_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load price data from Bloomberg or cache.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: yesterday)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with price data, indexed by date
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / f"prices_{start_date}_{end_date}.parquet"
        
        # Try loading from cache
        if use_cache and cache_file.exists():
            logger.info(f"📂 Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        
        # Load from Bloomberg
        if XBBG_AVAILABLE:
            df = self._load_from_bloomberg(start_date, end_date)
            if df is not None and not df.empty:
                # Save to cache
                df.to_parquet(cache_file)
                return df
        
        # Fallback to latest cache
        return self._load_fallback_cache()
    
    def _load_from_bloomberg(
        self,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load data from Bloomberg."""
        try:
            logger.info(f"🔌 Bloomberg 연결 시도 중... ({len(self.all_tickers)} tickers)")
            
            df = blp.bdh(
                tickers=self.all_tickers,
                flds=['px_last'],
                start_date=start_date,
                end_date=end_date,
                Per='D',
                Fill='NA'
            )
            
            if df is None or df.empty:
                raise ValueError("데이터가 비어있습니다.")
            
            # Clean column names (remove MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            df.index = pd.to_datetime(df.index)
            df = df.ffill().dropna()
            
            logger.info(f"✅ Bloomberg 데이터 로드 완료: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Bloomberg 로드 실패: {e}")
            return None
    
    def _load_fallback_cache(self) -> pd.DataFrame:
        """Load most recent cached data as fallback."""
        cache_files = list(self.cache_dir.glob("prices_*.parquet"))
        
        if cache_files:
            latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"📂 Fallback: Loading {latest_cache}")
            return pd.read_parquet(latest_cache)
        
        logger.warning("⚠️ No cached data available. Creating sample data.")
        return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing without Bloomberg."""
        import numpy as np
        
        dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='B')
        
        data = {}
        np.random.seed(42)
        
        # Sample rates data (yields)
        base_yields = {
            'USGG2YR Index': 2.0,
            'USGG10YR Index': 3.5,
            'GDBR2 Index': 0.5,
            'GDBR10 Index': 1.5,
            'GUKG10 Index': 2.5,
            'GJGB10 Index': 0.5,
            'GVSK3YR Index': 2.5,
            'GVSK10YR Index': 3.0,
            'GTAUD3YR Corp': 2.0,
            'GTAUD10YR Corp': 3.0,
            'GFRN10 Index': 1.8,
            'GBTPGR10 Index': 2.5,
        }
        
        for ticker, base in base_yields.items():
            # Random walk with mean reversion
            returns = np.random.randn(len(dates)) * 0.05
            cumulative = np.cumsum(returns)
            mean_reversion = -0.01 * cumulative  # Mean reversion component
            data[ticker] = base + cumulative + mean_reversion
        
        # Sample FX data
        base_fx = {
            'EUR Curncy': 1.10,
            'GBP Curncy': 1.30,
            'JPY Curncy': 110.0,
            'AUD Curncy': 0.70,
            'KRW Curncy': 1200.0,
        }
        
        for ticker, base in base_fx.items():
            vol = 0.005 if 'JPY' not in ticker and 'KRW' not in ticker else 0.003
            returns = np.random.randn(len(dates)) * vol
            data[ticker] = base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(data, index=dates)
        
        # Save as sample cache
        sample_cache = self.cache_dir / "prices_sample.parquet"
        df.to_parquet(sample_cache)
        logger.info(f"✅ Sample data created: {df.shape}")
        
        return df
    
    def get_cross_asset_pairs(self) -> Dict[str, List[tuple]]:
        """Get cross-asset pairs from configuration."""
        return self.config.get('cross_asset_pairs', {})
    
    def get_tickers_by_category(self) -> Dict[str, List[str]]:
        """Get tickers organized by category (rates/fx)."""
        result = {'rates': [], 'fx': []}
        
        if 'rates' in self.config:
            for country, data in self.config['rates'].items():
                if 'tickers' in data:
                    result['rates'].extend(data['tickers'])
        
        if 'fx' in self.config:
            for currency, data in self.config['fx'].items():
                if 'ticker' in data:
                    result['fx'].append(data['ticker'])
        
        return result
