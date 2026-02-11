"""
Strategy Factory for Global Macro Trading

Manages strategy storage, retrieval, and lifecycle in JSON format.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Manage strategy storage and retrieval."""
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize strategy factory.
        
        Args:
            storage_dir: Directory for storing strategy JSON files
        """
        self.storage_dir = Path(storage_dir) if storage_dir else \
            Path(__file__).parent / 'strategies'
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.storage_dir / 'index.json'
        self._load_index()
    
    def _load_index(self):
        """Load strategy index file."""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {'strategies': {}, 'last_updated': None}
    
    def _save_index(self):
        """Save strategy index file."""
        self.index['last_updated'] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def _generate_strategy_id(self, strategy_config: Dict[str, Any]) -> str:
        """
        Generate unique strategy ID based on configuration.
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            Unique strategy ID
        """
        # Create hash from key fields
        key_fields = {
            'asset': strategy_config.get('asset'),
            'strategy_type': strategy_config.get('strategy_type'),
            'strategy_name': strategy_config.get('strategy_name'),
            'params': str(sorted(strategy_config.get('params', {}).items())),
            'related_asset': strategy_config.get('related_asset'),
        }
        
        hash_str = json.dumps(key_fields, sort_keys=True)
        hash_id = hashlib.md5(hash_str.encode()).hexdigest()[:8]
        
        asset_short = strategy_config.get('asset', 'UNK')[:6]
        strategy_short = strategy_config.get('strategy_name', 'UNK')[:4].upper()
        type_short = 'MOM' if strategy_config.get('strategy_type') == 'momentum' else 'MR'
        
        return f"{asset_short}_{type_short}_{strategy_short}_{hash_id}"
    
    def save_strategy(
        self,
        strategy_config: Dict[str, Any],
        performance: Dict[str, Any]
    ) -> str:
        """
        Save strategy to factory.
        
        Args:
            strategy_config: Strategy configuration dict
            performance: Performance metrics dict
            
        Returns:
            Strategy ID
        """
        strategy_id = strategy_config.get('id') or self._generate_strategy_id(strategy_config)
        
        strategy_data = {
            'strategy_id': strategy_id,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'asset': strategy_config.get('asset'),
            'strategy_type': strategy_config.get('strategy_type'),
            'strategy_name': strategy_config.get('strategy_name'),
            'params': strategy_config.get('params', {}),
            'related_asset': strategy_config.get('related_asset'),
            'performance': {
                'sharpe_3y': performance.get('sharpe_3y', 0),
                'sharpe_6m': performance.get('sharpe_6m', 0),
                'sharpe_1m': performance.get('sharpe_1m', 0),
                'total_return': performance.get('total_return', 0),
                'annual_return': performance.get('annual_return', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'win_rate': performance.get('win_rate', 0),
                'num_trades': performance.get('num_trades', 0),
                'calmar_ratio': performance.get('calmar_ratio', 0),
                'sortino_ratio': performance.get('sortino_ratio', 0),
            },
            'is_active': False,
            'version': 1,
        }
        
        # Save to file
        strategy_file = self.storage_dir / f"{strategy_id}.json"
        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(strategy_data, f, indent=2, ensure_ascii=False)
        
        # Update index
        self.index['strategies'][strategy_id] = {
            'file': str(strategy_file.name),
            'asset': strategy_data['asset'],
            'strategy_type': strategy_data['strategy_type'],
            'sharpe_3y': strategy_data['performance']['sharpe_3y'],
            'sharpe_6m': strategy_data['performance']['sharpe_6m'],
            'is_active': False,
            'updated_at': strategy_data['updated_at'],
        }
        self._save_index()
        
        logger.info(f"✅ Strategy saved: {strategy_id}")
        return strategy_id
    
    def load_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Load strategy by ID.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Strategy data dict or None
        """
        strategy_file = self.storage_dir / f"{strategy_id}.json"
        
        if not strategy_file.exists():
            logger.warning(f"Strategy not found: {strategy_id}")
            return None
        
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for strategy {strategy_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading strategy {strategy_id}: {e}")
            return None
    
    def update_performance(
        self,
        strategy_id: str,
        performance: Dict[str, Any]
    ) -> bool:
        """
        Update strategy performance metrics.
        
        Args:
            strategy_id: Strategy identifier
            performance: New performance metrics
            
        Returns:
            True if successful
        """
        strategy = self.load_strategy(strategy_id)
        if not strategy:
            return False
        
        # Ensure all performance metrics are JSON serializable
        clean_performance = {}
        for k, v in performance.items():
            if hasattr(v, 'tolist'):
                val = v.tolist()
                clean_performance[k] = val if len(val) > 1 else val[0]
            elif isinstance(v, (list, dict)):
                clean_performance[k] = v
            else:
                try:
                    clean_performance[k] = float(v)
                except (TypeError, ValueError):
                    clean_performance[k] = v
                    
        strategy['performance'].update(clean_performance)
        strategy['updated_at'] = datetime.now().isoformat()
        strategy['version'] = strategy.get('version', 1) + 1
        
        strategy_file = self.storage_dir / f"{strategy_id}.json"
        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=2, ensure_ascii=False)
        
        # Update index
        if strategy_id in self.index['strategies']:
            self.index['strategies'][strategy_id].update({
                'sharpe_3y': performance.get('sharpe_3y', 
                    self.index['strategies'][strategy_id]['sharpe_3y']),
                'sharpe_6m': performance.get('sharpe_6m',
                    self.index['strategies'][strategy_id]['sharpe_6m']),
                'updated_at': strategy['updated_at'],
            })
            self._save_index()
        
        return True
    
    def set_active(self, strategy_id: str, active: bool = True) -> bool:
        """
        Set strategy active status.
        
        Args:
            strategy_id: Strategy identifier
            active: Active status
            
        Returns:
            True if successful
        """
        strategy = self.load_strategy(strategy_id)
        if not strategy:
            return False
        
        strategy['is_active'] = active
        strategy['updated_at'] = datetime.now().isoformat()
        
        strategy_file = self.storage_dir / f"{strategy_id}.json"
        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=2, ensure_ascii=False)
        
        if strategy_id in self.index['strategies']:
            self.index['strategies'][strategy_id]['is_active'] = active
            self._save_index()
        
        return True
    
    def filter_by_sharpe_3y(self, min_sharpe: float = 0.8) -> List[Dict[str, Any]]:
        """
        Filter strategies by 3-year Sharpe ratio.
        
        Args:
            min_sharpe: Minimum Sharpe ratio
            
        Returns:
            List of qualifying strategies
        """
        result = []
        
        for strategy_id, info in self.index['strategies'].items():
            if info.get('sharpe_3y', 0) >= min_sharpe:
                strategy = self.load_strategy(strategy_id)
                if strategy:
                    result.append(strategy)
        
        logger.info(f"📊 Found {len(result)} strategies with Sharpe 3Y >= {min_sharpe}")
        return result
    
    def filter_by_sharpe_6m(self, min_sharpe: float = 0.9) -> List[Dict[str, Any]]:
        """
        Filter strategies by 6-month Sharpe ratio.
        
        Args:
            min_sharpe: Minimum Sharpe ratio
            
        Returns:
            List of qualifying strategies
        """
        result = []
        
        for strategy_id, info in self.index['strategies'].items():
            if info.get('sharpe_6m', 0) >= min_sharpe:
                strategy = self.load_strategy(strategy_id)
                if strategy:
                    result.append(strategy)
        
        logger.info(f"📊 Found {len(result)} strategies with Sharpe 6M >= {min_sharpe}")
        return result
    
    def get_active_strategies(self) -> List[Dict[str, Any]]:
        """
        Get all active strategies.
        
        Returns:
            List of active strategies
        """
        result = []
        
        for strategy_id, info in self.index['strategies'].items():
            if info.get('is_active', False):
                strategy = self.load_strategy(strategy_id)
                if strategy:
                    result.append(strategy)
        
        return result
    
    def activate_qualified_strategies(
        self,
        sharpe_6m_min: float = 0.9
    ) -> List[str]:
        """
        Activate strategies that meet 6-month Sharpe threshold.
        
        Args:
            sharpe_6m_min: Minimum 6-month Sharpe ratio
            
        Returns:
            List of activated strategy IDs
        """
        activated = []
        
        for strategy_id, info in self.index['strategies'].items():
            if info.get('sharpe_6m', 0) >= sharpe_6m_min:
                if not info.get('is_active', False):
                    self.set_active(strategy_id, True)
                    activated.append(strategy_id)
            else:
                # Deactivate if no longer qualifies
                if info.get('is_active', False):
                    self.set_active(strategy_id, False)
        
        logger.info(f"🚀 Activated {len(activated)} strategies (Sharpe 6M >= {sharpe_6m_min})")
        return activated
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get factory summary statistics.
        
        Returns:
            Summary dict
        """
        total = len(self.index['strategies'])
        active = sum(1 for s in self.index['strategies'].values() if s.get('is_active'))
        
        by_type = {'momentum': 0, 'mean_reversion': 0, 'advanced': 0}
        for info in self.index['strategies'].values():
            stype = info.get('strategy_type', 'unknown')
            if stype in by_type:
                by_type[stype] += 1
        
        return {
            'total_strategies': total,
            'active_strategies': active,
            'by_type': by_type,
            'last_updated': self.index.get('last_updated'),
        }
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """
        Delete strategy from factory.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            True if successful
        """
        strategy_file = self.storage_dir / f"{strategy_id}.json"
        
        if strategy_file.exists():
            strategy_file.unlink()
        
        if strategy_id in self.index['strategies']:
            del self.index['strategies'][strategy_id]
            self._save_index()
        
        logger.info(f"🗑️ Strategy deleted: {strategy_id}")
        return True
    
    def cleanup_underperformers(
        self,
        sharpe_3y_min: float = 0.5,
        max_age_days: int = 365
    ) -> int:
        """
        Remove strategies that consistently underperform.
        
        Args:
            sharpe_3y_min: Minimum Sharpe to keep
            max_age_days: Maximum age without updates
            
        Returns:
            Number of deleted strategies
        """
        deleted = 0
        to_delete = []
        
        for strategy_id, info in self.index['strategies'].items():
            # Low Sharpe
            if info.get('sharpe_3y', 0) < sharpe_3y_min:
                to_delete.append(strategy_id)
                continue
            
            # Old and inactive
            updated = info.get('updated_at')
            if updated:
                updated_date = datetime.fromisoformat(updated)
                age = (datetime.now() - updated_date).days
                if age > max_age_days and not info.get('is_active', False):
                    to_delete.append(strategy_id)
        
        for strategy_id in to_delete:
            self.delete_strategy(strategy_id)
            deleted += 1
        
        logger.info(f"🧹 Cleaned up {deleted} underperforming strategies")
        return deleted
