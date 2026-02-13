"""
Strategy Generator for Global Macro Trading

Automatically generates all possible strategy combinations from indicators and parameters.
"""

import itertools
import logging
from typing import Dict, Any, List, Generator, Optional
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class StrategyGenerator:
    """Generate all possible strategy combinations systematically."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize strategy generator.
        
        Args:
            config_path: Path to indicators.yaml configuration
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.assets_config = self._load_assets_config()
        
    def _get_default_config_path(self) -> str:
        """Get default path to indicators.yaml."""
        return str(Path(__file__).parent.parent.parent / 'config' / 'indicators.yaml')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load indicator configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
    
    def _load_assets_config(self) -> Dict[str, Any]:
        """Load assets configuration for carry pairs etc."""
        assets_path = Path(self.config_path).parent / 'assets.yaml'
        try:
            with open(assets_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}
    
    def _generate_param_combinations(
        self,
        param_config: Dict[str, List]
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate all parameter combinations.
        
        Args:
            param_config: Dict of param names to list of values
            
        Yields:
            Parameter combination dict
        """
        keys = list(param_config.keys())
        values = list(param_config.values())
        
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))
    
    def generate_momentum_strategies(
        self,
        assets: List[str],
        related_assets: Dict[str, List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate all momentum strategy combinations.
        
        Args:
            assets: List of asset tickers
            related_assets: Dict mapping asset to related assets
            
        Yields:
            Strategy configuration dict
        """
        tech_config = self.config.get('technical_indicators', {})
        
        # MA Crossover
        if 'sma_crossover' in tech_config:
            params = tech_config['sma_crossover']['params']
            for asset in assets:
                for fast in params['fast_period']:
                    for slow in params['slow_period']:
                        if fast < slow:  # Valid crossover only
                            yield {
                                'asset': asset,
                                'strategy_type': 'momentum',
                                'strategy_name': 'ma_crossover',
                                'params': {'fast_period': fast, 'slow_period': slow},
                                'related_asset': None
                            }
        
        # EMA Crossover
        if 'ema_crossover' in tech_config:
            params = tech_config['ema_crossover']['params']
            for asset in assets:
                for fast in params['fast_period']:
                    for slow in params['slow_period']:
                        if fast < slow:
                            yield {
                                'asset': asset,
                                'strategy_type': 'momentum',
                                'strategy_name': 'ema_crossover',
                                'params': {'fast_period': fast, 'slow_period': slow},
                                'related_asset': None
                            }
        
        # RSI Momentum
        if 'rsi' in tech_config:
            params = tech_config['rsi']['params']
            for asset in assets:
                for period in params['period']:
                    yield {
                        'asset': asset,
                        'strategy_type': 'momentum',
                        'strategy_name': 'rsi_momentum',
                        'params': {'period': period, 'entry_threshold': 50, 'exit_threshold': 50},
                        'related_asset': None
                    }
        
        # MACD Crossover
        if 'macd' in tech_config:
            params = tech_config['macd']['params']
            for asset in assets:
                for fast in params['fast_period']:
                    for slow in params['slow_period']:
                        for signal in params['signal_period']:
                            yield {
                                'asset': asset,
                                'strategy_type': 'momentum',
                                'strategy_name': 'macd_crossover',
                                'params': {
                                    'fast_period': fast,
                                    'slow_period': slow,
                                    'signal_period': signal
                                },
                                'related_asset': None
                            }
        
        # Rate of Change
        if 'rate_of_change' in tech_config:
            params = tech_config['rate_of_change']['params']
            for asset in assets:
                for period in params['period']:
                    yield {
                        'asset': asset,
                        'strategy_type': 'momentum',
                        'strategy_name': 'rate_of_change',
                        'params': {'period': period, 'entry_threshold': 0, 'exit_threshold': 0},
                        'related_asset': None
                    }
        
        # Spread Momentum (cross-asset)
        if related_assets:
            cross_config = self.config.get('cross_asset_indicators', {})
            if 'spread_zscore' in cross_config:
                periods = cross_config['spread_zscore']['params']['period']
                for asset, related_list in related_assets.items():
                    if asset not in assets:
                        continue
                    for related in related_list:
                        for period in periods:
                            yield {
                                'asset': asset,
                                'strategy_type': 'momentum',
                                'strategy_name': 'spread_momentum',
                                'params': {'period': period, 'threshold': 0},
                                'related_asset': related
                            }
        
        # Breakout (Donchian Channel)
        if 'breakout' in tech_config:
            params = tech_config['breakout']['params']
            for asset in assets:
                for period in params['period']:
                    yield {
                        'asset': asset,
                        'strategy_type': 'momentum',
                        'strategy_name': 'breakout',
                        'params': {'period': period},
                        'related_asset': None
                    }
    
    def generate_mean_reversion_strategies(
        self,
        assets: List[str],
        related_assets: Dict[str, List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate all mean reversion strategy combinations.
        
        Args:
            assets: List of asset tickers
            related_assets: Dict mapping asset to related assets
            
        Yields:
            Strategy configuration dict
        """
        tech_config = self.config.get('technical_indicators', {})
        cross_config = self.config.get('cross_asset_indicators', {})
        
        # Z-Score Reversion
        if 'zscore' in tech_config:
            params = tech_config['zscore']
            for asset in assets:
                for period in params['params']['period']:
                    for entry in params['thresholds']['entry']:
                        for exit_th in params['thresholds']['exit']:
                            yield {
                                'asset': asset,
                                'strategy_type': 'mean_reversion',
                                'strategy_name': 'zscore_reversion',
                                'params': {
                                    'period': period,
                                    'entry_threshold': entry,
                                    'exit_threshold': exit_th
                                },
                                'related_asset': None
                            }
        
        # RSI Extremes
        if 'rsi' in tech_config:
            params = tech_config['rsi']
            for asset in assets:
                for period in params['params']['period']:
                    for oversold in params['thresholds']['oversold']:
                        for overbought in params['thresholds']['overbought']:
                            yield {
                                'asset': asset,
                                'strategy_type': 'mean_reversion',
                                'strategy_name': 'rsi_extremes',
                                'params': {
                                    'period': period,
                                    'oversold': oversold,
                                    'overbought': overbought,
                                    'exit_level': 50
                                },
                                'related_asset': None
                            }
        
        # Bollinger Reversion
        if 'bollinger_bands' in tech_config:
            params = tech_config['bollinger_bands']['params']
            for asset in assets:
                for period in params['period']:
                    for std_dev in params['std_dev']:
                        yield {
                            'asset': asset,
                            'strategy_type': 'mean_reversion',
                            'strategy_name': 'bollinger_reversion',
                            'params': {'period': period, 'std_dev': std_dev},
                            'related_asset': None
                        }
        
        # Spread Z-Score Reversion
        if related_assets and 'spread_zscore' in cross_config:
            params = cross_config['spread_zscore']
            for asset, related_list in related_assets.items():
                if asset not in assets:
                    continue
                for related in related_list:
                    for period in params['params']['period']:
                        for entry in params['thresholds']['entry']:
                            for exit_th in params['thresholds']['exit']:
                                yield {
                                    'asset': asset,
                                    'strategy_type': 'mean_reversion',
                                    'strategy_name': 'spread_zscore_reversion',
                                    'params': {
                                        'period': period,
                                        'entry_threshold': entry,
                                        'exit_threshold': exit_th
                                    },
                                    'related_asset': related
                                }
        
        # Spread Percentile Reversion
        if related_assets and 'spread_percentile' in cross_config:
            params = cross_config['spread_percentile']
            for asset, related_list in related_assets.items():
                if asset not in assets:
                    continue
                for related in related_list:
                    for period in params['params']['period']:
                        for low in params['thresholds']['low']:
                            for high in params['thresholds']['high']:
                                yield {
                                    'asset': asset,
                                    'strategy_type': 'mean_reversion',
                                    'strategy_name': 'spread_percentile_reversion',
                                    'params': {
                                        'period': period,
                                        'low_percentile': low,
                                        'high_percentile': high,
                                        'exit_percentile': 50
                                    },
                                    'related_asset': related
                                }

    def generate_advanced_strategies(
        self,
        assets: List[str],
        related_assets: Dict[str, List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate all advanced strategy combinations."""
        adv_config = self.config.get('advanced_strategies', {
            'filtered_momentum': {'ma_period': [200], 'rsi_period': [14], 'rsi_entry': [50]},
            'lead_lag_momentum': {'period': [20], 'threshold': [0]},
            'multi_tf_momentum': {'periods': [[20, 60, 120], [10, 30, 60]]},
            'volatility_breakout': {'period': [20], 'k': [1.0, 1.5, 2.0]},
            'relative_strength_rank': {'period': [20], 'top_n': [2]}
        })

        # Filtered Momentum
        if 'filtered_momentum' in adv_config:
            p = adv_config['filtered_momentum']
            for asset in assets:
                for ma in p.get('ma_period', [200]):
                    for rsi_p in p.get('rsi_period', [14]):
                        for ent in p.get('rsi_entry', [50]):
                            yield {
                                'asset': asset,
                                'strategy_type': 'advanced',
                                'strategy_name': 'filtered_momentum',
                                'params': {'ma_period': ma, 'rsi_period': rsi_p, 'rsi_entry': ent, 'rsi_exit': ent},
                                'related_asset': None
                            }

        # Lead-Lag
        if 'lead_lag_momentum' in adv_config and related_assets:
            p = adv_config['lead_lag_momentum']
            for asset, related_list in related_assets.items():
                if asset not in assets:
                    continue
                for related in related_list:
                    for period in p.get('period', [20]):
                        for th in p.get('threshold', [0]):
                            yield {
                                'asset': asset,
                                'strategy_type': 'advanced',
                                'strategy_name': 'lead_lag_momentum',
                                'params': {'period': period, 'threshold': th},
                                'related_asset': related
                            }

        # Multi TF
        if 'multi_tf_momentum' in adv_config:
            p = adv_config['multi_tf_momentum']
            for asset in assets:
                for periods in p.get('periods', [[20, 60, 120]]):
                    yield {
                        'asset': asset,
                        'strategy_type': 'advanced',
                        'strategy_name': 'multi_tf_momentum',
                        'params': {'periods': periods},
                        'related_asset': None
                    }

        # Volatility Breakout
        if 'volatility_breakout' in adv_config:
            p = adv_config['volatility_breakout']
            for asset in assets:
                for period in p.get('period', [20]):
                    for k in p.get('k', [1.0, 1.5, 2.0]):
                        yield {
                            'asset': asset,
                            'strategy_type': 'advanced',
                            'strategy_name': 'volatility_breakout',
                            'params': {'period': period, 'k': k},
                            'related_asset': None
                        }

        # RS Rank
        if 'relative_strength_rank' in adv_config:
            p = adv_config['relative_strength_rank']
            for asset in assets:
                for period in p.get('period', [20]):
                    for n in p.get('top_n', [2]):
                        yield {
                            'asset': asset,
                            'strategy_type': 'advanced',
                            'strategy_name': 'relative_strength_rank',
                            'params': {'period': period, 'top_n': n},
                            'related_asset': None
                        }

        # ADX Momentum
        if 'adx_momentum' in adv_config:
            p = adv_config['adx_momentum']
            for asset in assets:
                for adx_p in p.get('adx_period', [14]):
                    for roc_p in p.get('roc_period', [10]):
                        for threshold in p.get('adx_threshold', [25]):
                            yield {
                                'asset': asset,
                                'strategy_type': 'advanced',
                                'strategy_name': 'adx_momentum',
                                'params': {
                                    'adx_period': adx_p,
                                    'roc_period': roc_p,
                                    'adx_threshold': threshold
                                },
                                'related_asset': None
                            }

        # Carry Trade
        if 'carry_trade' in adv_config:
            p = adv_config['carry_trade']
            carry_pairs = self.assets_config.get('carry_pairs', [])
            for pair in carry_pairs:
                fx = pair['fx']
                if fx not in assets:
                    continue
                for period in p.get('period', [20]):
                    for threshold in p.get('threshold', [1.0]):
                        yield {
                            'asset': fx,
                            'strategy_type': 'advanced',
                            'strategy_name': 'carry_trade',
                            'params': {
                                'rate_asset': pair['us_rate'],
                                'foreign_rate_asset': pair['foreign_rate'],
                                'period': period,
                                'threshold': threshold
                            },
                            'related_asset': pair['foreign_rate']
                        }
    
    def generate_all_strategies(
        self,
        assets: List[str],
        related_assets: Dict[str, List[str]] = None,
        sample_ratio: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate all strategy combinations (momentum + mean reversion).
        
        Args:
            assets: List of asset tickers
            related_assets: Dict mapping asset to related assets
            sample_ratio: Ratio of strategies to sample (0.0 to 1.0)
            
        Yields:
            Strategy configuration dict
        """
        import random
        strategy_id = 0
        
        # Momentum strategies
        for strategy in self.generate_momentum_strategies(assets, related_assets):
            if sample_ratio < 1.0 and random.random() > sample_ratio:
                continue
            strategy['id'] = f"MOM_{strategy_id:06d}"
            strategy_id += 1
            yield strategy
        
        # Mean reversion strategies
        for strategy in self.generate_mean_reversion_strategies(assets, related_assets):
            if sample_ratio < 1.0 and random.random() > sample_ratio:
                continue
            strategy['id'] = f"MR_{strategy_id:06d}"
            strategy_id += 1
            yield strategy

        # Advanced strategies
        for strategy in self.generate_advanced_strategies(assets, related_assets):
            if sample_ratio < 1.0 and random.random() > sample_ratio:
                continue
            strategy['id'] = f"ADV_{strategy_id:06d}"
            strategy_id += 1
            yield strategy
    
    def count_strategies(
        self,
        assets: List[str],
        related_assets: Dict[str, List[str]] = None
    ) -> Dict[str, int]:
        """
        Count total number of strategies to be generated.
        
        Args:
            assets: List of asset tickers
            related_assets: Dict mapping asset to related assets
            
        Returns:
            Dict with counts by strategy type
        """
        momentum_count = sum(1 for _ in self.generate_momentum_strategies(assets, related_assets))
        mean_rev_count = sum(1 for _ in self.generate_mean_reversion_strategies(assets, related_assets))
        advanced_count = sum(1 for _ in self.generate_advanced_strategies(assets, related_assets))
        
        return {
            'momentum': momentum_count,
            'mean_reversion': mean_rev_count,
            'advanced': advanced_count,
            'total': momentum_count + mean_rev_count + advanced_count
        }
