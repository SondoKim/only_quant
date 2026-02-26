"""
Global Macro Trading System - Main Pipeline

LLM-Free automated trading strategy discovery, backtesting, and signal generation.
"""

import argparse
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

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


def _resolve_signals_storage_dir(explicit_storage_dir: str = None) -> str:
    """
    Resolve the strategy storage directory for signals mode.

    If explicit_storage_dir is given, use it directly.
    Otherwise, auto-detect from today's date:
      - Today is in month M  →  use last business day of month M-1
      - Looks for 'strategies_YYYY-MM-DD' directories inside src/factory/
        and picks the newest one whose date falls within month M-1.
      - Falls back to 'strategies_present' if nothing is found.
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

    # Fallback
    fallback = factory_base / 'strategies_present'
    logger.warning(f"⚠️  [signals] No folder found for {prev_year}-{prev_month:02d}. "
                   f"Falling back to strategies_present.")
    return str(fallback)


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
                
                # Update selector threshold
                self.strategy_selector.sharpe_threshold = self.sharpe_6m_threshold
        else:
            self.sharpe_3y_threshold = 1.0
            self.sortino_3y_threshold = 1.0
            self.sharpe_6m_threshold = 0.5
            self.sortino_6m_threshold = 0.5
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
        backtester = VectorBTEngine(prices)
        
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
    
    def run_daily_update(self, max_correlation: float = 0.3, pivot_enabled: bool = None) -> Dict[str, Any]:
        """
        Run daily update pipeline.
        
        Updates performance metrics and generates trading signals.
        
        Returns:
            Daily update results
        """
        logger.info("📅 Running daily update...")
        
        if pivot_enabled is not None:
            if 'pivot_settings' not in self.strategy_selector.config:
                self.strategy_selector.config['pivot_settings'] = {}
            self.strategy_selector.config['pivot_settings']['enabled'] = pivot_enabled
            logger.info(f"⚙️ Pivot enabled overridden by caller: {pivot_enabled}")

        # 1. Load latest data
        prices = self.data_loader.load_data()
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()
        
        # 2. Update strategy performance
        backtester = VectorBTEngine(prices)
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
    
    def get_trading_signals(self, max_correlation: float = 0.3, pivot_enabled: bool = None) -> Dict[str, Any]:
        """
        Get current trading signals.
        
        Args:
            max_correlation: Maximum allowed correlation between strategies
            pivot_enabled: Optional override for pivot enabled setting
            
        Returns:
            Trading signal report
        """
        if pivot_enabled is not None:
            if 'pivot_settings' not in self.strategy_selector.config:
                self.strategy_selector.config['pivot_settings'] = {}
            self.strategy_selector.config['pivot_settings']['enabled'] = pivot_enabled
            logger.info(f"⚙️ Pivot enabled overridden by caller: {pivot_enabled}")

        prices = self.data_loader.load_data()
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()
        
        return self.strategy_selector.get_trading_report(prices, max_correlation=max_correlation)
    
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
    parser.add_argument('--pivot', dest='pivot', action='store_true', help='Force enable pivot logic')
    parser.add_argument('--no-pivot', dest='pivot', action='store_false', help='Force disable pivot logic')
    parser.set_defaults(pivot=None)
    
    args = parser.parse_args()
    
    # For signals mode, auto-resolve the storage dir if not explicitly provided
    storage_dir = args.storage_dir
    if args.mode == 'signals' and not storage_dir:
        storage_dir = _resolve_signals_storage_dir()

    system = GlobalMacroTradingSystem(storage_dir=storage_dir)
    
    if args.mode == 'discover':
        result = system.run_discovery(
            start_date=args.start_date,
            end_date=args.end_date,
            batch_size=args.batch_size,
            sample_ratio=args.sample_ratio,
            sample_count=args.sample_count,
            target_tickers=args.tickers
        )
        print("\n📊 Discovery Results:")
        for key, value in result.items():
            print(f"   {key}: {value}")
    
    elif args.mode == 'update':
        result = system.run_daily_update(max_correlation=args.max_corr, pivot_enabled=args.pivot)
        print("\n📅 Daily Update Results:")
        
        sharpe = result.get('portfolio_sharpe', 0.0)
        pivot_active = result.get('portfolio_pivot_active', False)
        pivot_status = "🚨 [PIVOT ACTIVE]" if pivot_active else "✅ [NORMAL]"
        print(f"   Portfolio Rolling Sharpe: {sharpe:.2f} {pivot_status}")
        if pivot_active:
            print("   ⚠️  All positions are CURRENTLY REVERSED due to trend exhaustion.")
        
        print(f"   Date: {result['date']}")
        print(f"   Active strategies: {result['total_active_strategies']}")
        positions = result['asset_positions']
        if not args.include_index:
            positions = [p for p in positions if 'NQ' not in p['asset']]
        
        # Per-active-asset budget allocation
        # Budget = 1,000만원 / active_count → max net = ±1,000만원 when all agree
        rate_keywords = ['Index', 'Corp']
        fx_keywords = ['Curncy']
        LONG_KEYWORDS = ['🟢 LONG']
        SHORT_KEYWORDS = ['🔴 SHORT']
        rate_positions = [p for p in positions if any(k in p['asset'] for k in rate_keywords)
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
        
        print("\n   📊 금리 포지션:")
        for pos in positions:
            if not any(k in pos['asset'] for k in rate_keywords):
                continue
            delta_str = f", 델타: {delta_map[pos['asset']]:+d}만원" if pos['asset'] in delta_map else ""
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(신뢰도: {pos['confidence']:.2f}, 전략개수: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}]{delta_str})")
        
        if rate_positions:
            long_cnt = sum(1 for p in rate_positions if p['position'] in LONG_KEYWORDS)
            short_cnt = sum(1 for p in rate_positions if p['position'] in SHORT_KEYWORDS)
            gross = sum(abs(v) for v in delta_map.values())
            net_sign = "롱" if net_delta > 0 else "숏" if net_delta < 0 else "중립"
            print(f"\n      💰 금리 넷 델타: {net_delta:+d}만원 {net_sign} (롱 {long_cnt} / 숏 {short_cnt}, 그로스: {gross}만원, 예산: {round(1000/len(rate_positions))}만/자산)")
        
        print("\n   💱 FX 포지션:")
        for pos in positions:
            if not any(k in pos['asset'] for k in fx_keywords):
                continue
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(신뢰도: {pos['confidence']:.2f}, 전략개수: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}])")
    
    elif args.mode == 'signals':
        result = system.get_trading_signals(max_correlation=args.max_corr, pivot_enabled=args.pivot)
        print("\n🎯 Trading Signals:")
        
        sharpe = result.get('portfolio_sharpe', 0.0)
        pivot_active = result.get('portfolio_pivot_active', False)
        pivot_status = "🚨 [PIVOT ACTIVE]" if pivot_active else "✅ [NORMAL]"
        print(f"   Portfolio Rolling Sharpe: {sharpe:.2f} {pivot_status}")
        if pivot_active:
            print("   ⚠️  All positions are CURRENTLY REVERSED due to trend exhaustion.")

        print(f"   Date: {result['date']}")
        print(f"   Active strategies: {result['total_active_strategies']}")
        positions = result['asset_positions']
        if not args.include_index:
            positions = [p for p in positions if 'NQ' not in p['asset']]
        
        # Per-active-asset budget allocation
        # For rates: display is INVERTED (yield direction → futures direction)
        rate_keywords = ['Index', 'Corp']
        fx_keywords = ['Curncy']
        LONG_KEYWORDS = ['🟢 LONG']
        SHORT_KEYWORDS = ['🔴 SHORT']
        # Helper: flip yield signal → futures direction for display
        def rate_futures_display(pos_str):
            if pos_str == '🟢 LONG':  return '🔴 SHORT'  # yield up → futures short
            if pos_str == '🔴 SHORT': return '🟢 LONG'   # yield down → futures long
            return pos_str
        rate_positions = [p for p in positions if any(k in p['asset'] for k in rate_keywords)
                         and p['confidence'] > 0 and p['position'] not in ['🔘 NOSTR', '⚪ FLAT']]
        delta_map = {}
        net_delta = 0
        if rate_positions:
            per_budget = 1000 / len(rate_positions)
            for p in rate_positions:
                # Futures delta: yield LONG → short futures → negative delta
                sign = -1 if p['position'] in LONG_KEYWORDS else 1
                d = round(p['confidence'] * per_budget * sign)
                delta_map[p['asset']] = d
                net_delta += d
        
        print("\n   📊 금리 포지션 (국채선물 기준):")
        for pos in positions:
            if not any(k in pos['asset'] for k in rate_keywords):
                continue
            display_pos = rate_futures_display(pos['position']) if pos['position'] not in ['🔘 NOSTR', '⚪ FLAT'] else pos['position']
            delta_str = f", 델타: {delta_map[pos['asset']]:+d}만원" if pos['asset'] in delta_map else ""
            print(f"      {pos['asset']}: {display_pos} "
                  f"(신뢰도: {pos['confidence']:.2f}, 전략개수: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}]{delta_str})")
        
        if rate_positions:
            # Futures perspective: yield-LONG = futures-SHORT
            fut_long_cnt  = sum(1 for p in rate_positions if p['position'] in SHORT_KEYWORDS)  # yield short → fut long
            fut_short_cnt = sum(1 for p in rate_positions if p['position'] in LONG_KEYWORDS)   # yield long  → fut short
            gross = sum(abs(v) for v in delta_map.values())
            net_sign = "롱" if net_delta > 0 else "숏" if net_delta < 0 else "중립"
            print(f"\n      💰 선물 넷 델타: {net_delta:+d}만원 {net_sign} (선물롱 {fut_long_cnt} / 선물숏 {fut_short_cnt}, 그로스: {gross}만원, 예산: {round(1000/len(rate_positions))}만/자산)")

        
        print("\n   💱 FX 포지션:")
        for pos in positions:
            if not any(k in pos['asset'] for k in fx_keywords):
                continue
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(신뢰도: {pos['confidence']:.2f}, 전략개수: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}])")
        
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
