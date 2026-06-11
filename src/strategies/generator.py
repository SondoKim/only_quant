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
    
    def _signal_only_tickers(self) -> set:
        """Yield tickers from signal_yields — inputs only, never traded."""
        sy = (self.assets_config.get('signal_yields', {}) or {})
        t = set((sy.get('tradeable_yield_map', {}) or {}).values())
        t |= set((sy.get('fx_short_yield', {}) or {}).values())
        t |= set((sy.get('policy_rate_map', {}) or {}).values())
        for pair in (sy.get('curve_slope_map', {}) or {}).values():
            t |= set(pair)
        return t

    @staticmethod
    def _enabled_cells(scope: Dict[str, Any]):
        """Parse discovery.enabled_cells → {asset_class: set(strategy_types)}.
        None = cell filtering disabled (allow everything)."""
        cells = scope.get('enabled_cells') or {}
        return {k: set(v) for k, v in cells.items()} if cells else None

    @staticmethod
    def _cell_ok(strategy: Dict[str, Any], enabled_cells) -> bool:
        """True if the strategy's asset-class × strategy_type cell is allowed.
        Keeps generation in sync with the selector's cell filter (2026-06-11
        honest-execution audit) so the dashboard universe and discovery search
        only contain investable cells."""
        if enabled_cells is None:
            return True
        a = strategy.get('asset', '')
        if 'Curncy' in a:
            klass = 'fx'
        elif 'NQ' in a or 'Index' in a:
            klass = 'index'
        elif 'Comdty' in a:
            klass = 'rates'
        else:
            klass = 'other'
        return strategy.get('strategy_type', '') in enabled_cells.get(klass, set())

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
    
    def generate_alpha_strategies(
        self,
        assets: List[str],
        related_assets: Dict[str, List[str]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate all alpha strategy combinations (alpha1/alpha2/alpha3)."""
        alpha_config = self.config.get('alpha_strategies', {})

        # ── Alpha1: Cross-Sectional Momentum ──
        if 'xsect_momentum' in alpha_config:
            p = alpha_config['xsect_momentum']
            for universe in p.get('universe', ['rates', 'fx']):
                if universe == 'rates':
                    target_assets = [a for a in assets if 'Comdty' in a and 'NQ' not in a]
                elif universe == 'fx':
                    target_assets = [a for a in assets if 'Curncy' in a]
                else:
                    target_assets = list(assets)
                for asset in target_assets:
                    for period in p.get('period', [20]):
                        for top_n in p.get('top_n', [3]):
                            yield {
                                'asset': asset,
                                'strategy_type': 'alpha1',
                                'strategy_name': 'xsect_momentum',
                                'params': {
                                    'period': period,
                                    'top_n': top_n,
                                    'universe': universe,
                                },
                                'related_asset': None,
                            }

        # ── Alpha1: Cross-Sectional Carry ──
        if 'xsect_carry' in alpha_config:
            p = alpha_config['xsect_carry']
            carry_pairs = self.assets_config.get('carry_pairs', [])
            for pair in carry_pairs:
                fx = pair['fx']
                if fx not in assets:
                    continue
                for period in p.get('period', [20]):
                    for top_n in p.get('top_n', [2]):
                        yield {
                            'asset': fx,
                            'strategy_type': 'alpha1',
                            'strategy_name': 'xsect_carry',
                            'params': {
                                'rate_asset': pair['us_rate'],
                                'foreign_rate_asset': pair['foreign_rate'],
                                'period': period,
                                'top_n': top_n,
                            },
                            'related_asset': pair['foreign_rate'],
                        }

        # ── Alpha2: Curve → FX ──
        if 'curve_to_fx' in alpha_config:
            p = alpha_config['curve_to_fx']
            pred_pairs = self.assets_config.get('predictive_pairs', {}).get('curve_to_fx', [])
            for pair in pred_pairs:
                short_rate = pair['short_rate']
                long_rate = pair['long_rate']
                for target in pair.get('targets', []):
                    if target not in assets:
                        continue
                    for period in p.get('period', [20]):
                        for threshold in p.get('threshold', [1.0]):
                            yield {
                                'asset': target,
                                'strategy_type': 'alpha2',
                                'strategy_name': 'curve_to_fx',
                                'params': {
                                    'short_rate': short_rate,
                                    'long_rate': long_rate,
                                    'period': period,
                                    'threshold': threshold,
                                },
                                'related_asset': long_rate,
                            }

        # ── Alpha2: Rate-Diff → FX ──
        if 'rate_diff_to_fx' in alpha_config:
            p = alpha_config['rate_diff_to_fx']
            pred_pairs = self.assets_config.get('predictive_pairs', {}).get('rate_diff_to_fx', [])
            for pair in pred_pairs:
                target = pair['target']
                if target not in assets:
                    continue
                for period in p.get('period', [20]):
                    for threshold in p.get('threshold', [0.5]):
                        yield {
                            'asset': target,
                            'strategy_type': 'alpha2',
                            'strategy_name': 'rate_diff_to_fx',
                            'params': {
                                'us_rate': pair['us_rate'],
                                'foreign_rate': pair['foreign_rate'],
                                'period': period,
                                'threshold': threshold,
                            },
                            'related_asset': pair['foreign_rate'],
                        }

        # ── Alpha4: Yield-based Carry / Value (uses merged signal_yields cols) ──
        sy = (self.assets_config.get('signal_yields', {}) or {})

        if 'rates_carry' in alpha_config:
            p = alpha_config['rates_carry']
            for asset, pair in (sy.get('curve_slope_map', {}) or {}).items():
                if asset not in assets or len(pair) != 2:
                    continue
                for threshold in p.get('threshold', [0.0]):
                    for smooth in p.get('smooth', [5]):
                        yield {
                            'asset': asset,
                            'strategy_type': 'alpha4',
                            'strategy_name': 'rates_carry',
                            'params': {
                                'y_short': pair[0], 'y_long': pair[1],
                                'threshold': threshold, 'smooth': smooth,
                            },
                            'related_asset': None,
                        }

        if 'rates_value' in alpha_config:
            p = alpha_config['rates_value']
            for asset, y_own in (sy.get('tradeable_yield_map', {}) or {}).items():
                if asset not in assets:
                    continue
                for lookback in p.get('lookback', [252]):
                    for entry_z in p.get('entry_z', [1.0]):
                        yield {
                            'asset': asset,
                            'strategy_type': 'alpha4',
                            'strategy_name': 'rates_value',
                            'params': {
                                'y_own': y_own,
                                'lookback': lookback, 'entry_z': entry_z,
                            },
                            'related_asset': None,
                        }

        if 'real_rate_fx' in alpha_config:
            p = alpha_config['real_rate_fx']
            fxmap = (sy.get('fx_short_yield', {}) or {})
            y_us = fxmap.get('US')
            for fx, y_foreign in fxmap.items():
                if fx == 'US' or fx not in assets or not y_us:
                    continue
                # KRW Curncy is USDKRW spot (inverted quote): KRW strength = down
                quote_sign = -1 if fx == 'KRW Curncy' else 1
                for period in p.get('period', [20]):
                    yield {
                        'asset': fx,
                        'strategy_type': 'alpha4',
                        'strategy_name': 'real_rate_fx',
                        'params': {
                            'y_foreign': y_foreign, 'y_us': y_us,
                            'period': period, 'quote_sign': quote_sign,
                        },
                        'related_asset': None,
                    }

        if 'policy_momentum' in alpha_config:
            p = alpha_config['policy_momentum']
            for asset, y_policy in (sy.get('policy_rate_map', {}) or {}).items():
                if asset not in assets:
                    continue
                for period in p.get('period', [60]):
                    yield {
                        'asset': asset,
                        'strategy_type': 'alpha4',
                        'strategy_name': 'policy_momentum',
                        'params': {'y_policy': y_policy, 'period': period},
                        'related_asset': None,
                    }

        if 'month_end_seasonal' in alpha_config:
            p = alpha_config['month_end_seasonal']
            # Documented for bonds (index duration extension) - rates futures only
            for asset in assets:
                if 'Comdty' not in asset:
                    continue
                for days_before in p.get('days_before', [3]):
                    yield {
                        'asset': asset,
                        'strategy_type': 'alpha4',
                        'strategy_name': 'month_end_seasonal',
                        'params': {'days_before': days_before},
                        'related_asset': None,
                    }


    def generate_all_strategies(
        self,
        assets: List[str],
        related_assets: Dict[str, List[str]] = None,
        sample_ratio: float = 1.0
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate all strategy combinations.

        Args:
            assets: List of asset tickers
            related_assets: Dict mapping asset to related assets
            sample_ratio: Ratio of strategies to sample (0.0 to 1.0)

        Yields:
            Strategy configuration dict
        """
        import random
        strategy_id = 0

        scope = self.config.get('discovery', {}) or {}
        # Which categories to generate (config-driven; default = all four)
        enabled = set(
            scope.get(
                'enabled_categories',
                ['momentum', 'mean_reversion', 'advanced', 'alpha'],
            )
        )
        # Assets removed from the strategy search universe entirely; their
        # prices may still feed other strategies as related/basket inputs.
        excluded_assets = set(scope.get('exclude_assets', []) or [])
        excluded_assets |= self._signal_only_tickers()
        # Strategy archetypes disabled for persistent live underperformance.
        disabled_names = set(scope.get('disabled_strategies', []) or [])
        enabled_cells = self._enabled_cells(scope)

        if excluded_assets:
            assets = [a for a in assets if a not in excluded_assets]

        def _skip(strategy: Dict[str, Any]) -> bool:
            return (strategy.get('asset') in excluded_assets
                    or strategy.get('strategy_name') in disabled_names
                    or not self._cell_ok(strategy, enabled_cells))

        # Momentum strategies
        if 'momentum' in enabled:
            for strategy in self.generate_momentum_strategies(assets, related_assets):
                if _skip(strategy):
                    continue
                if sample_ratio < 1.0 and random.random() > sample_ratio:
                    continue
                strategy['id'] = f"MOM_{strategy_id:06d}"
                strategy_id += 1
                yield strategy

        # Mean reversion strategies
        if 'mean_reversion' in enabled:
            for strategy in self.generate_mean_reversion_strategies(assets, related_assets):
                if _skip(strategy):
                    continue
                if sample_ratio < 1.0 and random.random() > sample_ratio:
                    continue
                strategy['id'] = f"MR_{strategy_id:06d}"
                strategy_id += 1
                yield strategy

        # Advanced strategies
        if 'advanced' in enabled:
            for strategy in self.generate_advanced_strategies(assets, related_assets):
                if _skip(strategy):
                    continue
                if sample_ratio < 1.0 and random.random() > sample_ratio:
                    continue
                strategy['id'] = f"ADV_{strategy_id:06d}"
                strategy_id += 1
                yield strategy

        # Alpha strategies (alpha1, alpha2, alpha3)
        if 'alpha' in enabled:
            for strategy in self.generate_alpha_strategies(assets, related_assets):
                if _skip(strategy):
                    continue
                if sample_ratio < 1.0 and random.random() > sample_ratio:
                    continue
                prefix = strategy['strategy_type'].upper()  # ALPHA1, ALPHA2, ALPHA3
                strategy['id'] = f"{prefix}_{strategy_id:06d}"
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
        scope = self.config.get('discovery', {}) or {}
        excluded_assets = set(scope.get('exclude_assets', []) or [])
        excluded_assets |= self._signal_only_tickers()
        disabled_names = set(scope.get('disabled_strategies', []) or [])
        enabled_cells = self._enabled_cells(scope)
        if excluded_assets:
            assets = [a for a in assets if a not in excluded_assets]

        def _ok(s: Dict[str, Any]) -> bool:
            return (s.get('asset') not in excluded_assets
                    and s.get('strategy_name') not in disabled_names
                    and self._cell_ok(s, enabled_cells))

        momentum_count = sum(1 for s in self.generate_momentum_strategies(assets, related_assets) if _ok(s))
        mean_rev_count = sum(1 for s in self.generate_mean_reversion_strategies(assets, related_assets) if _ok(s))
        advanced_count = sum(1 for s in self.generate_advanced_strategies(assets, related_assets) if _ok(s))

        alpha_counts = {'alpha1': 0, 'alpha2': 0, 'alpha3': 0, 'alpha4': 0}
        for s in self.generate_alpha_strategies(assets, related_assets):
            if not _ok(s):
                continue
            stype = s.get('strategy_type', '')
            if stype in alpha_counts:
                alpha_counts[stype] += 1
        alpha_total = sum(alpha_counts.values())

        return {
            'momentum': momentum_count,
            'mean_reversion': mean_rev_count,
            'advanced': advanced_count,
            'alpha1': alpha_counts['alpha1'],
            'alpha2': alpha_counts['alpha2'],
            'alpha3': alpha_counts['alpha3'],
            'alpha4': alpha_counts['alpha4'],
            'total': momentum_count + mean_rev_count + advanced_count + alpha_total,
        }
