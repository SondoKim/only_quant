"""
Global Macro Trading System - Main Pipeline

LLM-Free automated trading strategy discovery, backtesting, and signal generation.
"""

import argparse
import logging
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.strategies.generator import StrategyGenerator
from src.backtester.vectorbt_engine import VectorBTEngine
from src.factory.strategy_factory import StrategyFactory
from src.portfolio.selector import StrategySelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_last_business_day_of_prev_month() -> 'datetime.date':
    """
    Return the last business day (Mon-Fri) of the previous month.
    Does not account for public holidays.
    """
    from datetime import timedelta

    today = datetime.now().date()
    # Last day of previous month = day before the 1st of this month
    last_day = today.replace(day=1) - timedelta(days=1)
    # Walk backwards from last_day until we hit a weekday
    while last_day.weekday() >= 5:          # 5=Sat, 6=Sun
        last_day -= timedelta(days=1)
    return last_day


def _resolve_discover_settings(
    explicit_end_date: str = None,
    explicit_storage_dir: str = None,
) -> tuple:
    """
    Resolve end_date and storage_dir for discover mode.

    Returns:
        (end_date: str, storage_dir: str)
    """
    last_bday = _get_last_business_day_of_prev_month()
    end_date = explicit_end_date or last_bday.strftime('%Y-%m-%d')

    if explicit_storage_dir:
        storage_dir = explicit_storage_dir
    else:
        factory_base = Path(__file__).parent / 'src' / 'factory'
        storage_dir = str(factory_base / f'strategies_{end_date}')

    logger.info(f"📂 [discover] end_date  = {end_date}")
    logger.info(f"📂 [discover] storage   = {storage_dir}")
    return end_date, storage_dir


def _resolve_signals_storage_dir(explicit_storage_dir: str = None) -> str:
    """
    Resolve the strategy storage directory for signals mode.

    If explicit_storage_dir is given, use it directly.
    Otherwise, auto-detect from today's date:
      - Today is in month M  →  use last business day of month M-1
      - Looks for 'strategies_YYYY-MM-DD' directories inside src/factory/
        and picks the newest one whose date falls within month M-1.
      - Raises FileNotFoundError if nothing is found.
    """
    if explicit_storage_dir:
        return explicit_storage_dir

    import re
    from datetime import timedelta

    today = datetime.now().date()

    # Previous month
    first_of_this_month = today.replace(day=1)
    last_month = first_of_this_month - timedelta(days=1)
    prev_year  = last_month.year
    prev_month = last_month.month

    factory_base = Path(__file__).parent / 'src' / 'factory'

    # Find all strategies_YYYY-MM-DD directories
    pattern = re.compile(r'^strategies_(\d{4})-(\d{2})-(\d{2})$')
    candidates = []
    for d in factory_base.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if not m:
            continue
        y, mo, dy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Only consider folders from the previous month
        if y == prev_year and mo == prev_month:
            candidates.append((y, mo, dy, d))

    if candidates:
        # Pick the latest date within that month (= last business day)
        candidates.sort()
        chosen = candidates[-1][3]
        logger.info(f"📂 [signals] Auto-selected strategy folder: {chosen.name}")
        return str(chosen)

    # No folder found → error
    raise FileNotFoundError(
        f"No strategies folder found for {prev_year}-{prev_month:02d}. "
        f"Run 'python main.py --mode discover' first to generate one."
    )


def _find_previous_factory_dir(current_storage_dir: str) -> Optional[str]:
    """Find the most recent strategy folder BEFORE the current one.
    Used by incremental discovery to load previous month's strategies."""
    current = Path(current_storage_dir)
    factory_base = current.parent
    current_name = current.name

    pattern = re.compile(r'^strategies_(\d{4}-\d{2}-\d{2})$')
    candidates = []
    for d in factory_base.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m and d.name < current_name:
            if (d / 'strategies.db').exists() or (d / 'index.json').exists():
                candidates.append(d)

    if not candidates:
        return None
    candidates.sort(key=lambda x: x.name)
    return str(candidates[-1])


def _should_full_scan(current_end_date: str, full_scan_interval_months: int = 3) -> bool:
    """Determine if this month requires a full scan based on interval.
    Full scan on months where (month number) % interval == 0."""
    dt = datetime.strptime(current_end_date, '%Y-%m-%d')
    return dt.month % full_scan_interval_months == 0


class GlobalMacroTradingSystem:
    """Main orchestrator for the trading system."""

    def __init__(self, config_dir: str = None, storage_dir: str = None):
        """
        Initialize trading system.
        
        Args:
            config_dir: Path to configuration directory
            storage_dir: Path to strategy storage directory
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / 'config'
        
        # Initialize components
        self.data_loader = DataLoader(
            config_path=str(self.config_dir / 'assets.yaml')
        )
        self.strategy_generator = StrategyGenerator(
            config_path=str(self.config_dir / 'indicators.yaml')
        )
        self.strategy_factory = StrategyFactory(storage_dir=storage_dir)
        self.strategy_selector = StrategySelector(factory=self.strategy_factory)
        
        # Load configurations
        self._load_backtest_config()
    
    def _load_backtest_config(self):
        """Load backtest configuration."""
        indicators_config = self.config_dir / 'indicators.yaml'
        if indicators_config.exists():
            with open(indicators_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                bt_config = config.get('backtest', {})
                self.sharpe_3y_threshold = bt_config.get('sharpe_threshold_3y', 1.0)
                self.sortino_3y_threshold = bt_config.get('sortino_threshold_3y', 1.0)
                self.sharpe_6m_threshold = bt_config.get('sharpe_threshold_6m', 0.5)
                self.sortino_6m_threshold = bt_config.get('sortino_threshold_6m', 0.5)
                self.min_trades = bt_config.get('min_trades', 20)
                self.max_drawdown = bt_config.get('max_drawdown', -0.20)
                self.trail_stop_pct = bt_config.get('trail_stop_pct', 0)
                self.max_hold_days = bt_config.get('max_hold_days', 0)
                
                # Update selector threshold
                self.strategy_selector.sharpe_threshold = self.sharpe_6m_threshold
        else:
            self.sharpe_3y_threshold = 0.8
            self.sortino_3y_threshold = 0.8
            self.sharpe_6m_threshold = 0.5
            self.sortino_6m_threshold = 0.5
            self.trail_stop_pct = 0
            self.max_hold_days = 0
            self.min_trades = 20
            self.max_drawdown = -0.20
    
    def run_discovery(
        self,
        start_date: str = "2020-01-01",
        end_date: str = None,
        batch_size: int = 1000,
        sample_ratio: float = 1.0,
        sample_count: int = None,
        target_tickers: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run strategy discovery pipeline.
        
        Discovers new strategies, backtests them, and stores qualified ones.
        
        Args:
            start_date: Data start date
            batch_size: Number of strategies to process per batch
            sample_ratio: Ratio of strategies to sample (0.0 to 1.0)
            sample_count: (Alternative to ratio) Target number of strategies to test
            target_tickers: Optional list of tickers to limit discovery to
            
        Returns:
            Discovery results summary
        """
        logger.info("🚀 Starting strategy discovery pipeline...")
        
        # 1. Load data
        logger.info(f"📊 Loading price data from {start_date} to {end_date or 'latest'}...")
        prices = self.data_loader.load_data(start_date=start_date, end_date=end_date)
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()
        
        # Filter prices by target_tickers if provided
        if target_tickers:
            available_tickers = [t for t in target_tickers if t in prices.columns]
            if not available_tickers:
                logger.error(f"❌ None of the requested tickers {target_tickers} found in data.")
                return {'error': 'No valid tickers found'}
            prices = prices[available_tickers]
            logger.info(f"🎯 Filtered discovery to: {available_tickers}")
        
        logger.info(f"   Data shape: {prices.shape}")
        logger.info(f"   Date range: {prices.index[0]} to {prices.index[-1]}")
        
        # 2. Get assets and related assets
        assets = list(prices.columns)
        related_assets = self._build_related_assets_map()
        
        # 3. Count strategies
        counts = self.strategy_generator.count_strategies(assets, related_assets)
        total_possible = counts['total']
        
        # Adjust sample_ratio if sample_count is provided
        if sample_count is not None and sample_count > 0:
            sample_ratio = min(1.0, sample_count / total_possible)
            logger.info(f"🎯 Target sample count: {sample_count} (Ratio set to: {sample_ratio:.4f})")
        
        logger.info(f"📈 Total possible strategies: {total_possible}")
        if sample_ratio < 1.0:
            logger.info(f"🔄 Sampling mode: {sample_ratio*100:.1f}% search")
        
        # 4. Initialize backtester
        backtester = VectorBTEngine(
            prices,
            trail_stop_pct=self.trail_stop_pct,
            max_hold_days=self.max_hold_days,
        )
        
        # 5. Generate and backtest strategies in batches
        tested_count = 0
        stored_count = 0
        active_count = 0
        batch = []
        
        generator = self.strategy_generator.generate_all_strategies(
            assets, related_assets, sample_ratio=sample_ratio
        )
        
        for strategy in generator:
            batch.append(strategy)
            tested_count += 1
            
            if len(batch) >= batch_size:
                stored, active = self._process_batch(batch, backtester)
                stored_count += stored
                active_count += active
                batch = []
        
        # Process remaining
        if batch:
            stored, active = self._process_batch(batch, backtester)
            stored_count += stored
            active_count += active
        
        # 6. Summary
        summary = {
            'total_tested': tested_count,
            'total_possible': total_possible,
            'stored_strategies': stored_count,
            'active_strategies': active_count,
            'discovery_rate': (tested_count / total_possible) if total_possible > 0 else 0,
            'storage_rate': (stored_count / tested_count) if tested_count > 0 else 0,
        }
        
        logger.info("✅ Discovery complete!")
        logger.info(f"   Tested: {tested_count}/{total_possible}")
        logger.info(f"   Stored: {stored_count} (Sharpe 3Y >= {self.sharpe_3y_threshold}, Sortino 3Y >= {self.sortino_3y_threshold})")

        return summary

    def run_incremental_discovery(
        self,
        start_date: str = "2020-01-01",
        end_date: str = None,
        batch_size: int = 1000,
        new_sample_ratio: float = 0.1,
        near_threshold_band: float = 0.2,
        target_tickers: List[str] = None,
        full_scan_interval: int = 3,
    ) -> Dict[str, Any]:
        """
        Incremental strategy discovery.

        1. Re-backtest previous month's stored strategies with new data window
        2. Full re-test strategies near the threshold (sharpe within band)
        3. Sample-test remaining new strategies
        4. Every full_scan_interval months, run full scan instead

        Args:
            start_date: Data start date
            end_date: Data end date
            batch_size: Batch size for processing
            new_sample_ratio: Ratio of new strategies to sample (0.0 to 1.0)
            near_threshold_band: Re-test strategies with sharpe_3y in
                                 [threshold - band, threshold). e.g. 0.2 means 0.8~1.0
            target_tickers: Optional ticker filter
            full_scan_interval: Full scan every N months (default 3)
        """
        # Check if full scan is needed (every N months)
        if end_date and _should_full_scan(end_date, full_scan_interval):
            logger.info(f"📅 Full scan month (every {full_scan_interval} months). Running full discovery.")
            return self.run_discovery(
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                sample_ratio=1.0,
                target_tickers=target_tickers,
            )

        # Find previous month's factory
        prev_dir = _find_previous_factory_dir(str(self.strategy_factory.storage_dir))
        if prev_dir is None:
            logger.info("📂 No previous factory found. Running full discovery.")
            return self.run_discovery(
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                sample_ratio=1.0,
                target_tickers=target_tickers,
            )

        logger.info(f"🔄 Incremental discovery (prev: {Path(prev_dir).name})")

        # 1. Load data
        prices = self.data_loader.load_data(start_date=start_date, end_date=end_date)
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()

        if target_tickers:
            available_tickers = [t for t in target_tickers if t in prices.columns]
            if not available_tickers:
                return {'error': 'No valid tickers found'}
            prices = prices[available_tickers]

        assets = list(prices.columns)
        related_assets = self._build_related_assets_map()
        backtester = VectorBTEngine(
            prices,
            trail_stop_pct=self.trail_stop_pct,
            max_hold_days=self.max_hold_days,
        )

        # 2. Load previous month's strategies
        prev_factory = StrategyFactory(storage_dir=prev_dir)
        prev_configs = prev_factory.get_all_strategy_configs()
        prev_factory.close()

        logger.info(f"   Previous month: {len(prev_configs)} strategies loaded")

        # Separate into: (a) previously passed, (b) near-threshold candidates
        prev_passed = [c for c in prev_configs
                       if c.get('_prev_sharpe_3y', 0) >= self.sharpe_3y_threshold]
        near_threshold = [c for c in prev_configs
                          if self.sharpe_3y_threshold - near_threshold_band
                          <= c.get('_prev_sharpe_3y', 0)
                          < self.sharpe_3y_threshold]

        # 3. Re-backtest previous winners
        tested_count = 0
        stored_count = 0
        active_count = 0

        logger.info(f"   Phase A: Re-testing {len(prev_passed)} previous winners...")
        for i in range(0, len(prev_passed), batch_size):
            batch = prev_passed[i:i + batch_size]
            stored, active = self._process_batch(batch, backtester)
            tested_count += len(batch)
            stored_count += stored
            active_count += active

        # 4. Re-test near-threshold strategies
        if near_threshold:
            logger.info(f"   Phase B: Re-testing {len(near_threshold)} near-threshold strategies...")
            for i in range(0, len(near_threshold), batch_size):
                batch = near_threshold[i:i + batch_size]
                stored, active = self._process_batch(batch, backtester)
                tested_count += len(batch)
                stored_count += stored
                active_count += active

        # 5. Sample new strategies (not in previous month's DB)
        prev_ids = {c['id'] for c in prev_configs}
        logger.info(f"   Phase C: Sampling new strategies (ratio: {new_sample_ratio:.0%})...")

        new_batch = []
        new_tested = 0
        generator = self.strategy_generator.generate_all_strategies(
            assets, related_assets, sample_ratio=1.0
        )
        import random
        for strategy in generator:
            if strategy.get('id') in prev_ids:
                continue  # Skip already-tested strategies
            if random.random() > new_sample_ratio:
                continue
            new_batch.append(strategy)
            new_tested += 1

            if len(new_batch) >= batch_size:
                stored, active = self._process_batch(new_batch, backtester)
                tested_count += len(new_batch)
                stored_count += stored
                active_count += active
                new_batch = []

        if new_batch:
            stored, active = self._process_batch(new_batch, backtester)
            tested_count += len(new_batch)
            stored_count += stored
            active_count += active

        logger.info(f"   Phase C: {new_tested} new strategies sampled")

        # Summary
        total_possible = self.strategy_generator.count_strategies(assets, related_assets)['total']
        summary = {
            'mode': 'incremental',
            'total_tested': tested_count,
            'total_possible': total_possible,
            'prev_winners_retested': len(prev_passed),
            'near_threshold_retested': len(near_threshold),
            'new_sampled': new_tested,
            'stored_strategies': stored_count,
            'active_strategies': active_count,
            'discovery_rate': (tested_count / total_possible) if total_possible > 0 else 0,
            'storage_rate': (stored_count / tested_count) if tested_count > 0 else 0,
        }

        logger.info("✅ Incremental discovery complete!")
        logger.info(f"   Tested: {tested_count} (prev: {len(prev_passed)}, near: {len(near_threshold)}, new: {new_tested})")
        logger.info(f"   Stored: {stored_count}")

        return summary

    def _process_batch(
        self,
        strategies: List[Dict[str, Any]],
        backtester: VectorBTEngine
    ) -> tuple:
        """Process a batch of strategies."""
        # Run backtests
        results = backtester.run_batch_backtest(strategies, progress=True)
        
        # Filter and store
        stored = 0
        active = 0
        
        for strategy, result in zip(strategies, results):
            # Check if qualifies for storage: Sharpe 3Y AND Sortino 3Y
            if (result.sharpe_3y >= self.sharpe_3y_threshold and
                result.sortino_ratio >= self.sortino_3y_threshold and
                result.num_trades >= self.min_trades and
                result.max_drawdown >= self.max_drawdown):
                
                self.strategy_factory.save_strategy(strategy, result.to_dict())
                stored += 1
                
                self.strategy_factory.set_active(
                    strategy.get('id') or self.strategy_factory._generate_strategy_id(strategy),
                    True
                )
                active += 1
        
        return stored, active
    
    def _build_related_assets_map(self) -> Dict[str, List[str]]:
        """Build map of related assets from config."""
        cross_pairs = self.data_loader.get_cross_asset_pairs()
        related = {}
        
        for category, pairs in cross_pairs.items():
            for pair in pairs:
                if len(pair) == 2:
                    asset1, asset2 = pair
                    
                    if asset1 not in related:
                        related[asset1] = []
                    if asset2 not in related[asset1]:
                        related[asset1].append(asset2)
                    
                    if asset2 not in related:
                        related[asset2] = []
                    if asset1 not in related[asset2]:
                        related[asset2].append(asset1)
        
        return related
    
    def run_daily_update(self, max_correlation: float = 0.3) -> Dict[str, Any]:
        """
        Run daily update pipeline.

        Updates performance metrics and generates trading signals.

        Returns:
            Daily update results
        """
        logger.info("📅 Running daily update...")

        # 1. Load latest data
        prices = self.data_loader.load_data()
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()
        
        # 2. Update strategy performance
        backtester = VectorBTEngine(
            prices,
            trail_stop_pct=self.trail_stop_pct,
            max_hold_days=self.max_hold_days,
        )
        updated = 0
        
        for strategy in self.strategy_factory.filter_by_sharpe_3y(0):  # Get all
            result = backtester.run_backtest(strategy)
            self.strategy_factory.update_performance(
                strategy['strategy_id'],
                result.to_dict()
            )
            updated += 1
        
        logger.info(f"   Updated {updated} strategy performances")
        
        # 4. Generate trading signals
        report = self.strategy_selector.get_trading_report(prices, max_correlation=max_correlation)
        
        logger.info("✅ Daily update complete!")
        return report
    
    def get_trading_signals(self, max_correlation: float = 0.3,
                            hurst_window: int = 120, hurst_threshold: float = 0.5,
                            hurst_method: str = 'rs') -> Dict[str, Any]:
        """
        Get current trading signals with Hurst Exponent regime classification.

        Args:
            max_correlation: Maximum allowed correlation between strategies
            hurst_window: Lookback window for Hurst Exponent (default: 120)
            hurst_threshold: H threshold above which regime is 'trending' (default: 0.5)

        Returns:
            Trading signal report including 'regime_info' per asset
        """
        prices = self.data_loader.load_data()
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()

        result = self.strategy_selector.get_trading_report(prices, max_correlation=max_correlation)

        # Compute Hurst Exponent regime for each asset
        from src.indicators.technical import TechnicalIndicators
        regime_info: Dict[str, Any] = {}
        for asset in prices.columns:
            h = TechnicalIndicators.hurst(prices[asset], window=hurst_window, method=hurst_method)
            regime_info[asset] = {
                'hurst': round(h, 3),
                'regime': 'trending' if h > hurst_threshold else 'ranging',
                'method': hurst_method.upper(),
            }
        result['regime_info'] = regime_info

        return result
    
    def get_factory_summary(self) -> Dict[str, Any]:
        """Get strategy factory summary."""
        return self.strategy_factory.get_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Global Macro Trading System')
    parser.add_argument('--mode', choices=['discover', 'update', 'signals', 'summary'],
                       default='signals', help='Execution mode')
    parser.add_argument('--start-date', default='2020-01-01',
                       help='Data start date for discovery')
    parser.add_argument('--end-date', default=None,
                       help='Data end date for discovery')
    parser.add_argument('--storage-dir', default=None,
                       help='Directory for strategy storage')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for strategy processing')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                       help='Ratio of strategies to sample (0.0 to 1.0)')
    parser.add_argument('--sample-count', type=int, default=None,
                        help='Target number of strategies to test (overrides sample-ratio)')
    parser.add_argument('--max-corr', type=float, default=0.5,
                        help='Maximum allowed correlation between strategies (0.0 to 1.0)')
    parser.add_argument('--tickers', nargs='+', 
                        help='Specific tickers to process (e.g. "NQ1 Index" "USGG10YR Index")')
    parser.add_argument('--include-index', action='store_true',
                         help='Include index assets (e.g. NQ1 Index) in signal output')
    parser.add_argument('--verbose', action='store_true',
                         help='Show per-strategy details for each asset in signals/update mode')
    parser.add_argument('--incremental', action='store_true',
                         help='Use incremental discovery (reuse previous month strategies)')
    parser.add_argument('--full-scan-interval', type=int, default=3,
                         help='Full scan every N months in incremental mode (default: 3)')
    parser.add_argument('--new-sample-ratio', type=float, default=0.1,
                         help='Ratio of new strategies to sample in incremental mode (default: 0.1)')
    args = parser.parse_args()
    
    # Auto-resolve storage_dir / end_date depending on mode
    storage_dir = args.storage_dir
    end_date = args.end_date

    if args.mode == 'discover':
        end_date, storage_dir = _resolve_discover_settings(
            explicit_end_date=args.end_date,
            explicit_storage_dir=args.storage_dir,
        )
    elif args.mode == 'signals' and not storage_dir:
        storage_dir = _resolve_signals_storage_dir()

    system = GlobalMacroTradingSystem(storage_dir=storage_dir)
    
    if args.mode == 'discover':
        if args.incremental:
            result = system.run_incremental_discovery(
                start_date=args.start_date,
                end_date=end_date,
                batch_size=args.batch_size,
                new_sample_ratio=args.new_sample_ratio,
                target_tickers=args.tickers,
                full_scan_interval=args.full_scan_interval,
            )
        else:
            result = system.run_discovery(
                start_date=args.start_date,
                end_date=end_date,
                batch_size=args.batch_size,
                sample_ratio=args.sample_ratio,
                sample_count=args.sample_count,
                target_tickers=args.tickers,
            )
        print("\n📊 Discovery Results:")
        for key, value in result.items():
            print(f"   {key}: {value}")
    
    elif args.mode == 'update':
        result = system.run_daily_update(max_correlation=args.max_corr)
        print("\n📅 Daily Update Results:")
        print(f"   Date: {result['date']}")
        print(f"   Active strategies: {result['total_active_strategies']}")
        positions = result['asset_positions']
        if not args.include_index:
            positions = [p for p in positions if 'NQ' not in p['asset']]
        
        # Per-active-asset budget allocation (전체 선물 기준, 방향 역전 불필요)
        rate_keywords = ['Comdty']
        fx_keywords = ['Curncy']
        LONG_KEYWORDS = ['🟢 LONG']
        SHORT_KEYWORDS = ['🔴 SHORT']

        rate_positions = [p for p in positions if any(k in p['asset'] for k in rate_keywords)
                         and 'NQ' not in p['asset']
                         and p['confidence'] > 0 and p['position'] not in ['🔘 NOSTR', '⚪ FLAT']]
        delta_map = {}
        net_delta = 0
        if rate_positions:
            per_budget = 1000 / len(rate_positions)
            for p in rate_positions:
                sign = 1 if p['position'] in LONG_KEYWORDS else -1
                d = round(p['confidence'] * per_budget * sign)
                delta_map[p['asset']] = d
                net_delta += d

        print("\n   📊 금리 포지션 (선물 기준):")
        for pos in positions:
            if not any(k in pos['asset'] for k in rate_keywords):
                continue
            if 'NQ' in pos['asset']:
                continue
            delta_str = f", 델타: {delta_map[pos['asset']]:+d}만원" if pos['asset'] in delta_map else ""
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(신뢰도: {pos['confidence']:.2f}, 전략개수: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}]{delta_str})")

        if rate_positions:
            long_cnt  = sum(1 for p in rate_positions if p['position'] in LONG_KEYWORDS)
            short_cnt = sum(1 for p in rate_positions if p['position'] in SHORT_KEYWORDS)
            gross = sum(abs(v) for v in delta_map.values())
            net_sign = "롱" if net_delta > 0 else "숏" if net_delta < 0 else "중립"
            print(f"\n      💰 선물 넷 델타: {net_delta:+d}만원 {net_sign} (선물롱 {long_cnt} / 선물숏 {short_cnt}, 그로스: {gross}만원, 예산: {round(1000/len(rate_positions))}만/자산)")
        
        print("\n   💱 FX 포지션:")
        for pos in positions:
            if not any(k in pos['asset'] for k in fx_keywords):
                continue
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(신뢰도: {pos['confidence']:.2f}, 전략개수: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}])")
    
    elif args.mode == 'signals':
        result = system.get_trading_signals(max_correlation=args.max_corr)
        print("\n🎯 Trading Signals:")
        print(f"   Date: {result['date']}")
        print(f"   Active strategies: {result['total_active_strategies']}")
        positions = result['asset_positions']
        if not args.include_index:
            positions = [p for p in positions if 'NQ' not in p['asset']]

        # ── Hurst Exponent 시장 국면 출력 ────────────────────────────────
        regime_info = result.get('regime_info', {})
        if regime_info:
            sample_method = next(iter(regime_info.values()), {}).get('method', 'RS')
            print(f"\n   🌡️  시장 국면 (Hurst [{sample_method}], window={120}):")
            rate_assets_ri = sorted(
                [(a, ri) for a, ri in regime_info.items() if 'Comdty' in a and 'NQ' not in a],
                key=lambda x: x[0]
            )
            fx_assets_ri = sorted(
                [(a, ri) for a, ri in regime_info.items() if 'Curncy' in a],
                key=lambda x: x[0]
            )
            if rate_assets_ri:
                print("      [금리]")
                for asset, ri in rate_assets_ri:
                    icon = '📈 추세장' if ri['regime'] == 'trending' else '↔️  횡보장'
                    print(f"         {icon}  {asset}: H={ri['hurst']:.3f}")
            if fx_assets_ri:
                print("      [FX]")
                for asset, ri in fx_assets_ri:
                    icon = '📈 추세장' if ri['regime'] == 'trending' else '↔️  횡보장'
                    print(f"         {icon}  {asset}: H={ri['hurst']:.3f}")

        # Per-active-asset budget allocation (전체 선물 기준, 방향 역전 불필요)
        rate_keywords = ['Comdty']
        fx_keywords = ['Curncy']
        LONG_KEYWORDS = ['🟢 LONG']
        SHORT_KEYWORDS = ['🔴 SHORT']

        rate_positions = [p for p in positions if any(k in p['asset'] for k in rate_keywords)
                         and 'NQ' not in p['asset']
                         and p['confidence'] > 0 and p['position'] not in ['🔘 NOSTR', '⚪ FLAT']]
        delta_map = {}
        net_delta = 0
        if rate_positions:
            per_budget = 1000 / len(rate_positions)
            for p in rate_positions:
                sign = 1 if p['position'] in LONG_KEYWORDS else -1
                d = round(p['confidence'] * per_budget * sign)
                delta_map[p['asset']] = d
                net_delta += d

        print("\n   📊 금리 포지션 (선물 기준):")
        for pos in positions:
            if not any(k in pos['asset'] for k in rate_keywords):
                continue
            if 'NQ' in pos['asset']:
                continue
            delta_str = f", 델타: {delta_map[pos['asset']]:+d}만원" if pos['asset'] in delta_map else ""
            ri = regime_info.get(pos['asset'])
            regime_str = f" | {'📈' if ri['regime'] == 'trending' else '↔️'} H={ri['hurst']:.3f}" if ri else ""
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(신뢰도: {pos['confidence']:.2f}, 전략개수: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}]{delta_str}{regime_str})")

        if rate_positions:
            fut_long_cnt  = sum(1 for p in rate_positions if p['position'] in LONG_KEYWORDS)
            fut_short_cnt = sum(1 for p in rate_positions if p['position'] in SHORT_KEYWORDS)
            gross = sum(abs(v) for v in delta_map.values())
            net_sign = "롱" if net_delta > 0 else "숏" if net_delta < 0 else "중립"
            print(f"\n      💰 선물 넷 델타: {net_delta:+d}만원 {net_sign} (선물롱 {fut_long_cnt} / 선물숏 {fut_short_cnt}, 그로스: {gross}만원, 예산: {round(1000/len(rate_positions))}만/자산)")

        print("\n   💱 FX 포지션:")
        for pos in positions:
            if not any(k in pos['asset'] for k in fx_keywords):
                continue
            ri = regime_info.get(pos['asset'])
            regime_str = f" | {'📈' if ri['regime'] == 'trending' else '↔️'} H={ri['hurst']:.3f}" if ri else ""
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(신뢰도: {pos['confidence']:.2f}, 전략개수: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}]{regime_str})")
        
        # Verbose: per-strategy detail grouped by asset
        if args.verbose and result.get('raw_signals'):
            from collections import defaultdict
            by_asset = defaultdict(list)
            for s in result['raw_signals']:
                by_asset[s['asset']].append(s)
            print("\n   🔍 전략 세부 내역:")
            for asset in sorted(by_asset.keys()):
                strats = by_asset[asset]
                print(f"      [{asset}]")
                for s in sorted(strats, key=lambda x: -x['sharpe_6m']):
                    pos_icon = '🟢' if s['position'] == 1 else '🔴' if s['position'] == -1 else '⚪'
                    print(f"         {pos_icon} {s['strategy_id']}  {s['strategy_name']}  Sharpe6M: {s['sharpe_6m']:.2f}")
    
    
    elif args.mode == 'summary':
        result = system.get_factory_summary()
        print("\n📈 Strategy Factory Summary:")
        for key, value in result.items():
            print(f"   {key}: {value}")


if __name__ == '__main__':
    main()
