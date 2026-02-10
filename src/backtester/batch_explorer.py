"""
Batch Explorer for Global Macro Trading

Efficiently evaluates thousands of strategy combinations using vectorized operations.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import vectorbt as vbt

from ..strategies.generator import StrategyGenerator
from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.advanced import AdvancedStrategies

logger = logging.getLogger(__name__)

class BatchExplorer:
    """Evaluates multiple strategies in a single vectorized pass."""
    
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices
        self.generator = StrategyGenerator()
        
    def evaluate_all_strategies(
        self, 
        sample_ratio: float = 1.0,
        sample_count: int = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Evaluate all possible strategy combinations.
        
        Returns:
            Tuple of (DataFrame of daily returns, Dict of strat_id -> config)
        """
        assets = list(self.prices.columns)
        related_dict = {a: [x for x in assets if x != a] for a in assets}
        
        strategies = list(self.generator.generate_all_strategies(
            assets=assets,
            related_assets=related_dict,
            sample_ratio=sample_ratio
        ))
        
        if sample_count and sample_count < len(strategies):
            import random
            strategies = random.sample(strategies, sample_count)
            
        logger.info(f"🚀 Batch evaluating {len(strategies)} strategies...")
        
        all_returns = {}
        strat_configs = {}
        
        # We group by asset to minimize data slicing
        for asset in assets:
            asset_strategies = [s for s in strategies if s['asset'] == asset]
            if not asset_strategies:
                continue
                
            logger.info(f"   - {asset}: Evaluating {len(asset_strategies)} strategies...")
            
            # Get asset prices and clean them
            asset_prices = self.prices[asset].ffill().bfill()
            
            # Check for invalid prices
            if asset_prices.isnull().any() or not np.isfinite(asset_prices).all():
                logger.warning(f"⚠️ Asset {asset} has non-finite prices. Filling with 1e-6.")
                asset_prices = asset_prices.fillna(0).replace([np.inf, -np.inf], 0).clip(lower=1e-6)

            # 🚀 Fix for non-positive prices (Common in Yields/Macro data)
            # vectorbt requires prices > 0 for internal return calculations
            vbt_prices = asset_prices.copy()
            if (vbt_prices <= 0).any():
                # Apply same transformation as VectorBTEngine:
                vbt_prices = np.exp(vbt_prices / 10.0) 
            
            # Final safety check for vectorbt
            if (vbt_prices <= 0).any():
                vbt_prices = vbt_prices.clip(lower=1e-6)
            
            entries_list = []
            exits_list = []
            col_names = []
            
            for strat in asset_strategies:
                try:
                    s_type = strat['strategy_type']
                    s_name = strat['strategy_name']
                    params = strat['params']
                    r_asset = strat.get('related_asset')
                    
                    if s_type == 'momentum':
                        entries, exits = MomentumStrategy.generate_signals(
                            self.prices, asset, s_name, params, r_asset
                        )
                        entries_list.append(entries)
                        exits_list.append(exits)
                        
                    elif s_type == 'mean_reversion':
                        l_ent, l_ex, s_ent, s_ex = MeanReversionStrategy.generate_signals(
                            self.prices, asset, s_name, params, r_asset
                        )
                        entries_list.append(l_ent)
                        exits_list.append(l_ex)
                        
                    elif s_type == 'advanced':
                        entries, exits = AdvancedStrategies.generate_signals(
                            self.prices, asset, s_name, params, r_asset
                        )
                        entries_list.append(entries)
                        exits_list.append(exits)
                    
                    # Create a unique ID
                    strat_id = f"{s_type}_{s_name}_{asset}_{r_asset}_{params}"
                    col_names.append(strat_id)
                    strat_configs[strat_id] = strat
                    
                except Exception as e:
                    continue
            
            if not entries_list:
                continue
                
            entries_df = pd.concat(entries_list, axis=1)
            exits_df = pd.concat(exits_list, axis=1)
            entries_df.columns = col_names
            exits_df.columns = col_names
            
            pf = vbt.Portfolio.from_signals(
                vbt_prices,
                entries=entries_df,
                exits=exits_df,
                freq='D',
                init_cash=10000,
                fees=0.0002,
                slippage=0.0001
            )
            
            rets = pf.returns()
            for col in rets.columns:
                all_returns[col] = rets[col]
                
        return pd.DataFrame(all_returns), strat_configs
