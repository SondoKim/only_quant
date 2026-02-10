import logging
import os
import glob
import json
import sys
import yaml
from pathlib import Path

# Add project root to sys.path to allow importing src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.factory.strategy_factory import StrategyFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config_thresholds():
    """Load Sharpe thresholds from indicators.yaml."""
    config_path = project_root / 'config' / 'indicators.yaml'
    try:
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                backtest = config.get('backtest', {})
                s3y = backtest.get('sharpe_threshold_3y', 1.0)
                s6m = backtest.get('sharpe_threshold_6m', 1.2)
                logger.info(f"⚙️ Loaded thresholds from config: 3Y >= {s3y}, 6M >= {s6m}")
                return s3y, s6m
    except Exception as e:
        logger.warning(f"Failed to load config, using defaults (1.0/1.2): {e}")
    return 1.0, 1.2

def clean_corrupted_files():
    """Delete empty or corrupted JSON files first."""
    strategy_dir = project_root / 'src' / 'factory' / 'strategies'
    files = glob.glob(str(strategy_dir / '*.json'))
    deleted = 0
    for f in files:
        if f.endswith('index.json'):
            continue
        try:
            if os.path.getsize(f) == 0:
                os.remove(f)
                deleted += 1
                logger.info(f"🗑️ Deleted empty file: {f}")
                continue
            
            with open(f, 'r', encoding='utf-8') as file:
                json.load(file)
        except json.JSONDecodeError:
            os.remove(f)
            deleted += 1
            logger.info(f"🗑️ Deleted corrupted file: {f}")
    if deleted > 0:
        logger.info(f"✅ Cleaned {deleted} corrupted/empty files.")

def purge_low_quality_strategies(sharpe_3y_threshold=None, sharpe_6m_threshold=None):
    # 1. Clean corrupted files first to avoid factory crash
    clean_corrupted_files()

    # 2. Load thresholds if not provided
    if sharpe_3y_threshold is None or sharpe_6m_threshold is None:
        c3y, c6m = load_config_thresholds()
        sharpe_3y_threshold = sharpe_3y_threshold or c3y
        sharpe_6m_threshold = sharpe_6m_threshold or c6m

    # 3. Initialize factory
    try:
        factory = StrategyFactory()
    except Exception as e:
        logger.error(f"Failed to initialize factory: {e}")
        return

    strategies = factory.index.get('strategies', {})
    
    to_delete = []
    for sid, info in strategies.items():
        sharpe_3y = info.get('sharpe_3y', 0)
        sharpe_6m = info.get('sharpe_6m', 0)
        
        # Filter: Delete if EITHER 3Y < threshold OR 6M < threshold
        if sharpe_3y < sharpe_3y_threshold or sharpe_6m < sharpe_6m_threshold:
            to_delete.append(sid)
            
    logger.info(f"🔍 Found {len(to_delete)} strategies failing criteria (3Y < {sharpe_3y_threshold} or 6M < {sharpe_6m_threshold})")
    
    count = 0
    for sid in to_delete:
        try:
            factory.delete_strategy(sid)
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {sid}: {e}")
        
    logger.info(f"✅ Successfully purged {count} strategies.")
    
    # Reload index to verify
    factory._load_index()
    remaining = len(factory.index.get('strategies', {}))
    logger.info(f"📊 Remaining strategies in factory: {remaining}")

if __name__ == "__main__":
    purge_low_quality_strategies()
