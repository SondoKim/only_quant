"""
Alpha Strategies for Global Macro Trading

Three alpha strategy families:
- Alpha1 (Cross-Sectional Factor): rank assets within universe, long top / short bottom
- Alpha2 (Cross-Asset Predictive): use one asset class to predict another
- Alpha3 (Regime-Adaptive): detect trending/ranging regime, apply matching strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from ..indicators.technical import TechnicalIndicators
from ..indicators.cross_asset import CrossAssetIndicators


class AlphaStrategies:
    """Generate alpha trading signals. All return 4-tuple (long/short)."""

    STRATEGY_TYPES = [
        # Alpha1: Cross-Sectional
        'xsect_momentum',
        'xsect_carry',
        # Alpha2: Cross-Asset Predictive
        'curve_to_fx',
        'rate_diff_to_fx',
    ]

    # Map strategy name → alpha category
    ALPHA_CATEGORY = {
        'xsect_momentum': 'alpha1',
        'xsect_carry': 'alpha1',
        'curve_to_fx': 'alpha2',
        'rate_diff_to_fx': 'alpha2',
    }

    @staticmethod
    def _empty_signals(index):
        e = pd.Series(False, index=index)
        return e, e.copy(), e.copy(), e.copy()

    # ──────────────────────────────────────────────────────────────────
    # Alpha1: Cross-Sectional Factor
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def xsect_momentum(
        prices: pd.DataFrame,
        asset: str,
        period: int = 20,
        top_n: int = 3,
        universe: str = 'rates',
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Cross-sectional momentum: long top-N, short bottom-N by period return.
        Universe is filtered to rates (Comdty ex-NQ) or fx (Curncy).
        """
        if universe == 'rates':
            cols = [c for c in prices.columns if 'Comdty' in c and 'NQ' not in c]
        elif universe == 'fx':
            cols = [c for c in prices.columns if 'Curncy' in c]
        else:
            cols = list(prices.columns)

        if asset not in cols or len(cols) < top_n * 2:
            return AlphaStrategies._empty_signals(prices.index)

        returns = prices[cols].pct_change(period)
        ranks = returns.rank(axis=1, ascending=False)  # 1 = best
        asset_rank = ranks[asset]
        n_assets = len(cols)

        # Long: enters top N
        long_entries = (asset_rank <= top_n) & (asset_rank.shift(1) > top_n)
        long_exits = (asset_rank > top_n) & (asset_rank.shift(1) <= top_n)

        # Short: enters bottom N
        bottom_threshold = n_assets - top_n
        short_entries = (asset_rank > bottom_threshold) & (asset_rank.shift(1) <= bottom_threshold)
        short_exits = (asset_rank <= bottom_threshold) & (asset_rank.shift(1) > bottom_threshold)

        return long_entries, long_exits, short_entries, short_exits

    @staticmethod
    def xsect_carry(
        prices: pd.DataFrame,
        asset: str,
        rate_asset: str,
        foreign_rate_asset: str,
        period: int = 20,
        top_n: int = 2,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Cross-sectional carry for FX: rank by rate-differential z-score,
        long high-carry (z > 1), short low-carry (z < -1).
        """
        if (rate_asset not in prices.columns
                or foreign_rate_asset not in prices.columns):
            return AlphaStrategies._empty_signals(prices.index)

        # Carry proxy = domestic rate price - foreign rate price (futures)
        carry = prices[rate_asset] - prices[foreign_rate_asset]
        carry_z = TechnicalIndicators.zscore(carry, period)

        long_entries = (carry_z > 1.0) & (carry_z.shift(1) <= 1.0)
        long_exits = (carry_z < 0) & (carry_z.shift(1) >= 0)

        short_entries = (carry_z < -1.0) & (carry_z.shift(1) >= -1.0)
        short_exits = (carry_z > 0) & (carry_z.shift(1) <= 0)

        return long_entries, long_exits, short_entries, short_exits

    # ──────────────────────────────────────────────────────────────────
    # Alpha2: Cross-Asset Predictive Signals
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def curve_to_fx(
        prices: pd.DataFrame,
        target_asset: str,
        short_rate: str,
        long_rate: str,
        period: int = 20,
        threshold: float = 1.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Yield-curve slope change → FX direction.

        For bond futures: higher price = lower yield.
        curve = long_rate_price - short_rate_price
        curve rising → long-end outperforming → steepening → risk-on
          → long risk/commodity FX
        curve falling → flattening → risk-off → short risk FX
        """
        if short_rate not in prices.columns or long_rate not in prices.columns:
            return AlphaStrategies._empty_signals(prices.index)

        curve = prices[long_rate] - prices[short_rate]
        curve_z = TechnicalIndicators.zscore(curve, period)

        long_entries = (curve_z > threshold) & (curve_z.shift(1) <= threshold)
        long_exits = (curve_z < 0) & (curve_z.shift(1) >= 0)

        short_entries = (curve_z < -threshold) & (curve_z.shift(1) >= -threshold)
        short_exits = (curve_z > 0) & (curve_z.shift(1) <= 0)

        return long_entries, long_exits, short_entries, short_exits

    @staticmethod
    def rate_diff_to_fx(
        prices: pd.DataFrame,
        target_asset: str,
        us_rate: str,
        foreign_rate: str,
        period: int = 20,
        threshold: float = 0.5,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Rate-differential momentum → FX direction.

        diff = foreign_rate_price - us_rate_price
        diff rising → foreign yield dropping faster (price up) → foreign advantage
          → long foreign FX
        diff falling → USD advantage → short foreign FX
        """
        if us_rate not in prices.columns or foreign_rate not in prices.columns:
            return AlphaStrategies._empty_signals(prices.index)

        diff = prices[foreign_rate] - prices[us_rate]
        diff_mom = TechnicalIndicators.zscore(diff.diff(period).dropna(), period)
        # Re-index to match prices
        diff_mom = diff_mom.reindex(prices.index)

        long_entries = (diff_mom > threshold) & (diff_mom.shift(1) <= threshold)
        long_exits = (diff_mom < 0) & (diff_mom.shift(1) >= 0)

        short_entries = (diff_mom < -threshold) & (diff_mom.shift(1) >= -threshold)
        short_exits = (diff_mom > 0) & (diff_mom.shift(1) <= 0)

        return long_entries, long_exits, short_entries, short_exits

    # ──────────────────────────────────────────────────────────────────
    # Dispatcher
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def generate_signals(
        cls,
        prices: pd.DataFrame,
        asset: str,
        strategy_name: str,
        params: Dict[str, Any],
        related_asset: Optional[str] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generate alpha signals. Always returns 4-tuple (long/short)."""
        price_series = prices[asset]

        if strategy_name == 'xsect_momentum':
            return cls.xsect_momentum(prices, asset, **params)
        elif strategy_name == 'xsect_carry':
            return cls.xsect_carry(prices, asset, **params)
        elif strategy_name == 'curve_to_fx':
            return cls.curve_to_fx(prices, asset, **params)
        elif strategy_name == 'rate_diff_to_fx':
            return cls.rate_diff_to_fx(prices, asset, **params)
        else:
            raise ValueError(f"Unknown alpha strategy: {strategy_name}")
