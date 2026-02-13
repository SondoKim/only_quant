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
from src.backtester.portfolio_backtester import PortfolioBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    def get_trading_signals(self, max_correlation: float = 0.3) -> Dict[str, Any]:
        """
        Get current trading signals.
        
        Args:
            max_correlation: Maximum allowed correlation between strategies
            
        Returns:
            Trading signal report
        """
        prices = self.data_loader.load_data()
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()
        
        return self.strategy_selector.get_trading_report(prices, max_correlation=max_correlation)
    
    def get_factory_summary(self) -> Dict[str, Any]:
        """Get strategy factory summary."""
        return self.strategy_factory.get_summary()
    
    def run_portfolio_backtest(
        self,
        start_date: str = "2023-01-01",
        max_correlation: float = 0.3,
        realistic: bool = False
    ) -> Dict[str, Any]:
        """Run portfolio backtest with monthly rebalancing."""
        logger.info(f"📊 Running {'realistic ' if realistic else ''}portfolio backtest from {start_date}...")
        
        # 1. Load data
        prices = self.data_loader.load_data()
        preprocessor = DataPreprocessor(prices)
        prices = preprocessor.clean().get_data()
        
        # 2. Run simulation
        backtester = PortfolioBacktester(
            prices=prices,
            factory=self.strategy_factory,
            selector_threshold=self.sharpe_6m_threshold,
            max_correlation=max_correlation,
            realistic=realistic
        )
        
        return backtester.run_simulation(start_date=start_date)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Global Macro Trading System')
    parser.add_argument('--mode', choices=['discover', 'update', 'signals', 'summary', 'backtest-portfolio'],
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
    parser.add_argument('--max-corr', type=float, default=0.3,
                        help='Maximum allowed correlation between strategies (0.0 to 1.0)')
    parser.add_argument('--realistic', action='store_true',
                        help='Run realistic walk-forward backtest (no future bias)')
    parser.add_argument('--tickers', nargs='+', 
                        help='Specific tickers to process (e.g. "NQ1 Index" "USGG10YR Index")')
    
    args = parser.parse_args()
    
    system = GlobalMacroTradingSystem(storage_dir=args.storage_dir)
    
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
        result = system.run_daily_update(max_correlation=args.max_corr)
        print("\n📅 Daily Update Results:")
        print(f"   Date: {result['date']}")
        print(f"   Active strategies: {result['total_active_strategies']}")
        print("\n   Asset Positions:")
        for pos in result['asset_positions']:
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(confidence: {pos['confidence']:.2f}, strategies: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}])")
    
    elif args.mode == 'signals':
        result = system.get_trading_signals(max_correlation=args.max_corr)
        print("\n🎯 Trading Signals:")
        print(f"   Date: {result['date']}")
        print(f"   Active strategies: {result['total_active_strategies']}")
        print("\n   Asset Positions:")
        for pos in result['asset_positions']:
            print(f"      {pos['asset']}: {pos['position']} "
                  f"(confidence: {pos['confidence']:.2f}, strategies: {pos['strategies']} "
                  f"[MOM: {pos['momentum']}, MR: {pos['mean_reversion']}, ADV: {pos['advanced']}])")
    
    elif args.mode == 'backtest-portfolio':
        result = system.run_portfolio_backtest(
            start_date=args.start_date,
            max_correlation=args.max_corr,
            realistic=args.realistic
        )
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
        else:
            print("\n📈 Portfolio Backtest Results:")
            print(f"   Total Return: {result['total_return']*100:.2f}%")
            print(f"   Sharpe Ratio: {result['sharpe']:.2f}")
            print(f"   Max Drawdown: {result['max_drawdown']*100:.2f}%")
            print(f"\n✅ Plot saved as portfolio_backtest.png")
    
    elif args.mode == 'summary':
        result = system.get_factory_summary()
        print("\n📈 Strategy Factory Summary:")
        for key, value in result.items():
            print(f"   {key}: {value}")


if __name__ == '__main__':
    main()
