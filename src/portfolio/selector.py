"""
Portfolio Strategy Selector for Global Macro Trading

Selects active strategies and generates aggregated trading signals.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..factory.strategy_factory import StrategyFactory
from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.advanced import AdvancedStrategies
from ..strategies.alpha import AlphaStrategies
from ..backtester.vectorbt_engine import VectorBTEngine

logger = logging.getLogger(__name__)


class StrategySelector:
    """Select and manage active trading strategies."""
    
    def __init__(
        self,
        factory: StrategyFactory = None,
        config_path: str = None
    ):
        """
        Initialize strategy selector.
        
        Args:
            factory: Strategy factory instance
            config_path: Path to indicators.yaml
        """
        self.factory = factory
        if not self.factory:
            raise ValueError("StrategySelector requires a StrategyFactory instance.")
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.active_strategies: List[Dict[str, Any]] = []
        self._carry_positions: Dict[str, float] = {}  # asset → last non-zero position
        self._regime_inverted_ids: set = set()  # strategy_ids whose signals are inverted by regime filter

    def _get_default_config_path(self) -> str:
        """Get default path to indicators.yaml."""
        from pathlib import Path
        return str(Path(__file__).parent.parent.parent / 'config' / 'indicators.yaml')

    def _load_config(self) -> Dict[str, Any]:
        """Load indicator configuration."""
        import yaml
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    
    async def refresh_active_strategies(
        self,
        prices: pd.DataFrame,
        max_correlation: float = 0.3
    ) -> int:
        """
        Refresh list of active strategies based on performance and correlation.
        
        Args:
            prices: Price DataFrame for return calculation
            max_correlation: Maximum correlation allowed between strategies
            
        Returns:
            Number of active strategies
        """
        # 1. Get all potential active strategies (already activated in factory)
        potential_strategies = self.factory.get_active_strategies()
        
        if not potential_strategies:
            self.active_strategies = []
            return 0
            
        if max_correlation >= 1.0:
            self.active_strategies = potential_strategies
            logger.info(f"🎯 {len(self.active_strategies)} active strategies loaded (no corr filter)")
            return len(self.active_strategies)

        # 3. Filter by correlation
        logger.info(f"🔍 Filtering {len(potential_strategies)} potential strategies by correlation (threshold: {max_correlation})...")
        self.active_strategies = self._filter_by_correlation(potential_strategies, prices, max_correlation)
        self.active_strategies = self._ensure_min_assets(self.active_strategies, potential_strategies, min_assets=3)

        logger.info(f"🎯 {len(self.active_strategies)} active strategies loaded after correlation filtering")
        return len(self.active_strategies)

    def _ensure_min_assets(
        self,
        selected: List[Dict[str, Any]],
        all_candidates: List[Dict[str, Any]],
        min_assets: int = 3
    ) -> List[Dict[str, Any]]:
        """
        After correlation filtering, ensure at least min_assets unique assets
        are represented. Adds the best strategy (by sharpe_6m) per missing asset.
        """
        covered = {s['asset'] for s in selected}
        if len(covered) >= min_assets:
            return selected

        # Best uncovered strategy per asset, sorted by sharpe_6m descending
        asset_best: Dict[str, Dict[str, Any]] = {}
        for s in all_candidates:
            asset = s['asset']
            if asset in covered:
                continue
            sharpe = s['performance']['sharpe_6m']
            if asset not in asset_best or sharpe > asset_best[asset]['performance']['sharpe_6m']:
                asset_best[asset] = s

        extras = sorted(asset_best.values(), key=lambda x: x['performance']['sharpe_6m'], reverse=True)
        result = list(selected)
        for extra in extras:
            if len(covered) >= min_assets:
                break
            result.append(extra)
            covered.add(extra['asset'])
            logger.info(f"   ➕ Min-asset enforcement: added {extra['asset']} ({extra['strategy_id']}, Sharpe6M={extra['performance']['sharpe_6m']:.2f})")

        return result

    def _build_asset_groups(self) -> Dict[str, List[str]]:
        """Build country/asset group mapping from assets.yaml config.
        Returns dict: {group_name: [ticker1, ticker2, ...]}"""
        import yaml
        from pathlib import Path
        assets_path = Path(self.config_path).parent / 'assets.yaml'
        try:
            with open(assets_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return {}

        groups = {}
        # Rates: grouped by country
        for country, data in cfg.get('rates', {}).items():
            tickers = data.get('tickers', [])
            if tickers:
                groups[f'rates_{country}'] = tickers

        # FX: each currency is its own group
        for currency, data in cfg.get('fx', {}).items():
            ticker = data.get('ticker')
            if ticker:
                groups[f'fx_{currency}'] = [ticker]

        # Indices
        for name, data in cfg.get('indices', {}).items():
            ticker = data.get('ticker')
            if ticker:
                groups[f'index_{name}'] = [ticker]

        return groups

    def _filter_by_correlation(
        self,
        strategies: List[Dict[str, Any]],
        prices: pd.DataFrame,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Filter strategies by correlation within country/asset groups.
        Correlation filter applies within each group independently,
        so strategies from different countries don't compete with each other.
        """
        if not strategies:
            return []

        # Build asset → group mapping
        asset_groups = self._build_asset_groups()
        asset_to_group = {}
        for group_name, tickers in asset_groups.items():
            for ticker in tickers:
                asset_to_group[ticker] = group_name

        # Group strategies by their asset's country group
        from collections import defaultdict
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for s in strategies:
            group = asset_to_group.get(s['asset'], f'ungrouped_{s["asset"]}')
            grouped[group].append(s)

        backtester = VectorBTEngine(prices)

        # ── Pre-compute ALL returns in one parallel batch ──────────────────
        # run_batch_backtest uses mp.Pool internally (already parallelised).
        # This replaces the old per-strategy sequential run_backtest() calls.
        logger.info(f"   Batch backtesting {len(strategies)} strategies (parallel)...")
        batch_results = backtester.run_batch_backtest(strategies, progress=False)

        # Build cache: strategy_id → returns Series (None if invalid)
        returns_cache: Dict[str, Optional[pd.Series]] = {}
        for strat, result in zip(strategies, batch_results):
            sid = strat['strategy_id']
            r = result.returns if result is not None else None
            returns_cache[sid] = r if (r is not None and not r.empty and r.std() != 0) else None

        all_selected = []

        for group_name, group_strategies in grouped.items():
            # Sort by 6m Sharpe within group (descending)
            sorted_strats = sorted(
                group_strategies,
                key=lambda x: x['performance']['sharpe_6m'],
                reverse=True
            )

            selected_returns = []
            for strategy in sorted_strats:
                rets = returns_cache.get(strategy['strategy_id'])
                if rets is None:
                    continue

                # Check correlation with already selected strategies IN THIS GROUP
                is_redundant = False
                for other_rets in selected_returns:
                    corr = rets.corr(other_rets)
                    if abs(corr) >= threshold:
                        is_redundant = True
                        break

                if not is_redundant:
                    all_selected.append(strategy)
                    selected_returns.append(rets)

        before_diversify = len(all_selected)

        # ── Pass 2: Negative Correlation Seeking (diversifiers) ──
        selected_ids = {s['strategy_id'] for s in all_selected}
        if all_selected:
            # Build portfolio return series — reuse cache (no extra backtest)
            all_sel_returns = [returns_cache[s['strategy_id']]
                               for s in all_selected
                               if returns_cache.get(s['strategy_id']) is not None]

            if all_sel_returns:
                portfolio_rets = pd.concat(all_sel_returns, axis=1).mean(axis=1)

                # Check top-200 rejected strategies (by Sharpe) — all already cached
                rejected = [s for s in strategies if s['strategy_id'] not in selected_ids]
                rejected = sorted(rejected,
                                  key=lambda x: x['performance']['sharpe_6m'],
                                  reverse=True)[:200]

                diversifiers_added = 0
                for strategy in rejected:
                    if diversifiers_added >= 5:
                        break
                    rets = returns_cache.get(strategy['strategy_id'])
                    if rets is None:
                        continue
                    corr = rets.corr(portfolio_rets)
                    if corr < -0.1 and strategy['performance']['sharpe_6m'] > 0:
                        all_selected.append(strategy)
                        diversifiers_added += 1
                        logger.info(f"   + Diversifier: {strategy['strategy_id']} "
                                     f"(corr={corr:.2f}, Sharpe6M={strategy['performance']['sharpe_6m']:.2f})")

        logger.info(f"   Correlation filter: {len(strategies)} -> {before_diversify} "
                     f"+ {len(all_selected) - before_diversify} diversifiers "
                     f"= {len(all_selected)} (within {len(grouped)} groups)")
        return all_selected
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        target_date: str = None,
        max_correlation: float = 0.3
    ) -> pd.DataFrame:
        """
        Generate signals for all active strategies.
        
        Args:
            prices: DataFrame with price data
            target_date: Date to generate signals for (default: latest)
            max_correlation: Max correlation for strategy selection
            
        Returns:
            DataFrame with signals per strategy
        """
        if not self.active_strategies:
            # We need to run the refresh synchronously here if possible, 
            # or ensure it's called before. Since we changed it to async, 
            # we should be careful. Actually, let's keep it simple for now.
            # I'll make refresh_active_strategies synchronous to avoid complexity.
            # (Changed from prospective 'async' to synchronous)
            self._refresh_sync(prices, max_correlation)
        
        if not self.active_strategies:
            logger.warning("⚠️ No active strategies")
            return pd.DataFrame()
        
        signals = []
        
        for strategy in self.active_strategies:
            try:
                signal = self._generate_strategy_signal(prices, strategy, target_date)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Signal generation failed for {strategy['strategy_id']}: {e}")
        
        if not signals:
            return pd.DataFrame()
        
        return pd.DataFrame(signals)
    
    def _compute_position_series(
        self,
        prices: pd.DataFrame,
        strategy: Dict[str, Any],
    ) -> pd.Series:
        """Compute the full direction-corrected position series for a strategy.

        Separated from _generate_strategy_signal so that callers can pre-compute
        the full series once and look up individual dates cheaply.
        """
        asset = strategy['asset']
        strategy_type = strategy['strategy_type']
        strategy_name = strategy['strategy_name']
        params = strategy['params']
        related_asset = strategy.get('related_asset')

        if strategy_type == 'momentum':
            entries, exits = MomentumStrategy.generate_signals(
                prices, asset, strategy_name, params, related_asset
            )
            position = self._calculate_position_momentum(entries, exits)
        elif strategy_type == 'mean_reversion':
            long_entries, long_exits, short_entries, short_exits = \
                MeanReversionStrategy.generate_signals(
                    prices, asset, strategy_name, params, related_asset
                )
            position = self._calculate_position_mean_reversion(
                long_entries, long_exits, short_entries, short_exits
            )
        elif strategy_type == 'advanced':
            entries, exits = AdvancedStrategies.generate_signals(
                prices, asset, strategy_name, params, related_asset
            )
            position = self._calculate_position_momentum(entries, exits)
        elif strategy_type in ('alpha1', 'alpha2', 'alpha3'):
            long_entries, long_exits, short_entries, short_exits = \
                AlphaStrategies.generate_signals(
                    prices, asset, strategy_name, params, related_asset
                )
            position = self._calculate_position_mean_reversion(
                long_entries, long_exits, short_entries, short_exits
            )
        else:
            raise ValueError(f"Unknown strategy category: {strategy_type}")

        # 방향 보정 규칙:
        # 1) ADV rates: 역방향 (금리 선물에서 ADV 시그널은 contrarian 성격)
        # 2) alpha2 FX: 역방향 (curve_to_fx / rate_diff_to_fx는 채권 가격 기준으로
        #    계산되어 yield 기준 경제적 방향과 반대가 됨)
        # 3) regime_inverted: 추세장의 MR 전략 → 신호 반전으로 추세 추종화
        is_rates = 'Comdty' in asset and 'NQ' not in asset
        is_fx = 'Curncy' in asset

        if strategy_type == 'advanced' and is_rates:
            position = -position
        elif strategy_type == 'alpha2' and is_fx:
            position = -position

        sid = strategy.get('strategy_id', '')
        if sid and sid in self._regime_inverted_ids:
            position = -position

        return position

    def precompute_position_cache(
        self,
        prices: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Pre-compute position series for all active strategies.

        Call once after monthly rebalance. The returned cache maps
        strategy_id → {'positions': pd.Series, <signal metadata>}
        so that daily lookups are O(1) dict access instead of
        recomputing the full indicator series each day.
        """
        cache: Dict[str, Any] = {}
        for strategy in self.active_strategies:
            try:
                pos_series = self._compute_position_series(prices, strategy)
                cache[strategy['strategy_id']] = {
                    'positions': pos_series,
                    'asset': strategy['asset'],
                    'related_asset': strategy.get('related_asset'),
                    'strategy_type': strategy['strategy_type'],
                    'strategy_name': strategy['strategy_name'],
                    'sharpe_6m': strategy['performance']['sharpe_6m'],
                }
            except Exception as e:
                logger.error(f"Position cache failed for {strategy['strategy_id']}: {e}")
        return cache

    def _generate_strategy_signal(
        self,
        prices: pd.DataFrame,
        strategy: Dict[str, Any],
        target_date: str = None
    ) -> Dict[str, Any]:
        """Generate signal for a single strategy at a specific date."""
        position = self._compute_position_series(prices, strategy)

        if target_date:
            target_idx = pd.to_datetime(target_date)
            current_position = position.loc[target_idx] if target_idx in position.index else position.iloc[-1]
        else:
            current_position = position.iloc[-1]

        return {
            'strategy_id': strategy['strategy_id'],
            'asset': strategy['asset'],
            'related_asset': strategy.get('related_asset'),
            'strategy_type': strategy['strategy_type'],
            'strategy_name': strategy['strategy_name'],
            'position': float(current_position),
            'sharpe_6m': strategy['performance']['sharpe_6m'],
            'date': target_date or str(position.index[-1].date()),
        }
    
    def _calculate_position_momentum(
        self,
        entries: pd.Series,
        exits: pd.Series
    ) -> pd.Series:
        """Calculate position series for momentum strategy."""
        position = pd.Series(0, index=entries.index)
        current = 0
        
        for i in range(len(entries)):
            if entries.iloc[i] and current == 0:
                current = 1
            elif exits.iloc[i] and current == 1:
                current = 0
            position.iloc[i] = current
        
        return position
    
    def _calculate_position_mean_reversion(
        self,
        long_entries: pd.Series,
        long_exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series
    ) -> pd.Series:
        """Calculate position series for mean reversion strategy."""
        position = pd.Series(0, index=long_entries.index)
        current = 0
        
        for i in range(len(long_entries)):
            if long_entries.iloc[i] and current == 0:
                current = 1
            elif short_entries.iloc[i] and current == 0:
                current = -1
            elif long_exits.iloc[i] and current == 1:
                current = 0
            elif short_exits.iloc[i] and current == -1:
                current = 0
            position.iloc[i] = current
        
        return position
    
    def aggregate_positions(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        method: str = 'sharpe_weighted',
        min_assets: int = 3
    ) -> pd.DataFrame:
        """
        Aggregate signals into asset-level positions.
        
        Args:
            signals: DataFrame with strategy signals
            method: Aggregation method ('equal', 'sharpe_weighted', 'vote')
            
        Returns:
            DataFrame with asset positions
        """
        if signals.empty:
            return pd.DataFrame()
        
        # Group by asset
        asset_positions = []
        
        for asset in signals['asset'].unique():
            asset_signals = signals[signals['asset'] == asset]
            avg_position = 0.0
            
            if method == 'equal':
                # Simple average
                avg_position = asset_signals['position'].mean()
            
            elif method == 'sharpe_weighted':
                # Sharpe-weighted average
                weights = asset_signals['sharpe_6m']
                if weights.sum() > 0:
                    avg_position = np.average(
                        asset_signals['position'],
                        weights=weights
                    )
                else:
                    avg_position = asset_signals['position'].mean()
            
            elif method == 'vote':
                # Majority vote
                positions = asset_signals['positions'].values # Fix: positions should be from individual strategy status
                # Actually, aggregation usually works on the current proposed positions
                pos_vals = asset_signals['position'].values
                if np.sum(pos_vals > 0) > np.sum(pos_vals < 0):
                    avg_position = 1
                elif np.sum(pos_vals < 0) > np.sum(pos_vals > 0):
                    avg_position = -1
                else:
                    avg_position = 0
            
            else:
                avg_position = asset_signals['position'].mean()
            
            # Cap to [-1, 1] — confidence never exceeds 1.0
            avg_position = float(np.clip(avg_position, -1.0, 1.0))

            asset_positions.append({
                'asset': asset,
                'position': avg_position,
                'raw_position': avg_position,
                'num_strategies': len(asset_signals),
                'momentum_count': (asset_signals['strategy_type'] == 'momentum').sum(),
                'mr_count': (asset_signals['strategy_type'] == 'mean_reversion').sum(),
                'adv_count': (asset_signals['strategy_type'] == 'advanced').sum(),
                'avg_sharpe_6m': asset_signals['sharpe_6m'].mean(),
            })
        
        asset_df = pd.DataFrame(asset_positions)

        # --- Minimum asset enforcement with carry-forward ---
        # Update carry state: remember last non-zero position per asset
        for _, row in asset_df.iterrows():
            if abs(row['position']) > 0.01:
                self._carry_positions[row['asset']] = row['position']

        if min_assets > 0:
            nonzero_count = (asset_df['position'].abs() > 0.01).sum()
            if nonzero_count < min_assets:
                flat_mask = asset_df['position'].abs() <= 0.01
                flat_df = asset_df[flat_mask].copy()
                flat_df['_carry'] = flat_df['asset'].map(
                    lambda a: self._carry_positions.get(a, 0.0)
                )
                # Only fill from assets that have carry history, best Sharpe first
                fill_candidates = (
                    flat_df[flat_df['_carry'].abs() > 0.01]
                    .sort_values('avg_sharpe_6m', ascending=False)
                )
                needed = int(min_assets - nonzero_count)
                filled = 0
                for idx in fill_candidates.head(needed).index:
                    asset_df.at[idx, 'position'] = fill_candidates.at[idx, '_carry']
                    filled += 1
                if filled:
                    logger.debug(f"   📌 Carry-forward: filled {filled} flat asset(s) to meet min {min_assets}")

        return asset_df

    def get_trading_report(
        self,
        prices: pd.DataFrame,
        target_date: str = None,
        max_correlation: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading report.
        
        Args:
            prices: Price DataFrame
            target_date: Target date
            max_correlation: Max correlation for strategy selection
            
        Returns:
            Trading report dict
        """
        signals = self.generate_signals(prices, target_date, max_correlation)
        positions = self.aggregate_positions(signals, prices)
        
        # Create summary including ALL assets in the price data
        asset_summary = []
        # Custom sort for assets: US, DE, UK, AU, JP, KR rates, then FX
        def asset_sort_key(ticker: str) -> tuple:
            if ticker in ('TU1 Comdty', 'TY1 Comdty'): group = 1
            elif ticker in ('DU1 Comdty', 'RX1 Comdty'): group = 2
            elif ticker == 'G 1 Comdty': group = 3
            elif ticker in ('YM1 Comdty', 'XM1 Comdty'): group = 4
            elif ticker == 'JB1 Comdty': group = 5
            elif ticker in ('KE1 Comdty', 'KAA1 Comdty'): group = 6
            elif 'Comdty' in ticker: group = 7   # OAT1, IK1 등
            elif 'NQ' in ticker: group = 9
            elif 'Curncy' in ticker: group = 8
            elif 'Index' in ticker or 'Corp' in ticker: group = 7
            else: group = 10
            return (group, ticker)

        all_assets = sorted(list(prices.columns), key=asset_sort_key)
        
        # Map existing positions for quick lookup
        pos_lookup = {row['asset']: row for _, row in positions.iterrows()} if not positions.empty else {}
        
        for asset in all_assets:
            if asset in pos_lookup:
                row = pos_lookup[asset]
                pos_str = '🟢 LONG' if row['position'] > 0 else \
                          '🔴 SHORT' if row['position'] < 0 else '⚪ FLAT'
                asset_summary.append({
                    'asset': asset,
                    'position': pos_str,
                    'confidence': abs(row['raw_position']),
                    'strategies': row['num_strategies'],
                    'momentum': row['momentum_count'],
                    'mean_reversion': row['mr_count'],
                    'advanced': row['adv_count'],
                })
            else:
                # Default "NOSIG" for assets with no active strategies
                asset_summary.append({
                    'asset': asset,
                    'position': '🔘 NOSTR',
                    'confidence': 0.0,
                    'strategies': 0,
                    'momentum': 0,
                    'mean_reversion': 0,
                    'advanced': 0,
                })
        
        return {
            'date': target_date or str(datetime.now().date()),
            'total_active_strategies': len(self.active_strategies),
            'signals_generated': len(signals),
            'asset_positions': asset_summary,
            'raw_signals': signals.to_dict('records') if not signals.empty else [],
            'aggregated_positions': positions.to_dict('records') if not positions.empty else [],
        }
    
    def get_position_for_asset(
        self,
        prices: pd.DataFrame,
        asset: str,
        target_date: str = None
    ) -> Tuple[float, float]:
        """
        Get aggregated position for a specific asset.
        
        Args:
            prices: Price DataFrame
            asset: Asset ticker
            target_date: Target date
            
        Returns:
            Tuple of (position, confidence)
        """
        signals = self.generate_signals(prices, target_date)
        
        if signals.empty:
            return 0, 0.0
        
        asset_signals = signals[signals['asset'] == asset]
        
        if asset_signals.empty:
            return 0, 0.0
        
        positions = self.aggregate_positions(asset_signals)
        
        if positions.empty:
            return 0, 0.0
        
        row = positions.iloc[0]
        return float(row['position']), abs(row['position'])
    
    def _refresh_sync(self, prices: pd.DataFrame, max_correlation: float = 0.3):
        """Internal synchronous refresh."""
        self._regime_inverted_ids = set()  # reset; apply_regime_filter() will repopulate if called
        # All strategies are already activated — just get them
        potential_strategies = self.factory.get_active_strategies()

        if not potential_strategies:
            self.active_strategies = []
            return

        if max_correlation >= 1.0:
            self.active_strategies = potential_strategies
            return

        self.active_strategies = self._filter_by_correlation(potential_strategies, prices, max_correlation)
        self.active_strategies = self._ensure_min_assets(self.active_strategies, potential_strategies, min_assets=3)

    def apply_regime_filter(
        self,
        prices: pd.DataFrame,
        hurst_window: int = 120,
        hurst_threshold: float = 0.5,
        hurst_method: str = 'rs',
        regime_cache: Optional[Dict[str, pd.Series]] = None,
        as_of_date=None,
    ) -> None:
        """Alpha3 meta-layer: classify market regime per asset via Hurst Exponent.

        H > hurst_threshold → trending (persistent):
            MOM/ADV 유지, MR 신호 반전 → 추세 추종으로 활용
        H ≤ hurst_threshold → ranging (anti-persistent):
            MR 정방향 유지, MOM/ADV 제거
        alpha1 / alpha2 는 regime-agnostic → 항상 유지.

        regime_cache: {asset: pd.Series(index=dates, values=H)} 사전 계산값.
                      as_of_date 기준으로 조회하여 매월 재계산 비용 제거.
        """
        from ..indicators.technical import TechnicalIndicators

        self._regime_inverted_ids = set()

        before = len(self.active_strategies)
        filtered = []
        regime_log: Dict[str, str] = {}

        for strategy in self.active_strategies:
            asset = strategy['asset']
            stype = strategy['strategy_type']
            sid   = strategy['strategy_id']

            if stype in ('alpha1', 'alpha2'):
                filtered.append(strategy)
                continue

            if regime_cache is None and asset not in prices.columns:
                filtered.append(strategy)
                continue

            if asset not in regime_log:
                if regime_cache is not None and asset in regime_cache:
                    # 캐시 조회: as_of_date 이전의 마지막 Hurst 값
                    h_series = regime_cache[asset]
                    if as_of_date is not None:
                        h_series = h_series[h_series.index <= as_of_date]
                    h = float(h_series.iloc[-1]) if not h_series.empty else 0.5
                else:
                    # 캐시 없을 때 직접 계산
                    h = TechnicalIndicators.hurst(
                        prices[asset], window=hurst_window, method=hurst_method
                    )
                regime_log[asset] = 'trending' if h > hurst_threshold else 'ranging'

            regime = regime_log[asset]

            if regime == 'trending':
                if stype in ('momentum', 'advanced'):
                    filtered.append(strategy)
                elif stype == 'mean_reversion':
                    filtered.append(strategy)
                    self._regime_inverted_ids.add(sid)  # 추세장 MR → 반전
            else:  # ranging
                if stype == 'mean_reversion':
                    filtered.append(strategy)
                # MOM/ADV 횡보장 제거

        all_candidates = self.active_strategies
        self.active_strategies = filtered
        self.active_strategies = self._ensure_min_assets(
            self.active_strategies, all_candidates, min_assets=3
        )

        after = len(self.active_strategies)
        n_inverted = len(self._regime_inverted_ids)
        trending_assets = [a for a, r in regime_log.items() if r == 'trending']
        ranging_assets  = [a for a, r in regime_log.items() if r == 'ranging']
        logger.info(
            f"   Regime filter ({hurst_method.upper()}, H>{hurst_threshold}, window={hurst_window}): "
            f"{before} → {after} strategies "
            f"(MR inverted in trending: {n_inverted}) | "
            f"trending={len(trending_assets)} assets, ranging={len(ranging_assets)} assets"
        )
