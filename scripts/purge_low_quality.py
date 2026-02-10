from src.factory.strategy_factory import StrategyFactory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def purge_low_quality_strategies(threshold: float = 1.0):
    factory = StrategyFactory()
    strategies = factory.index.get('strategies', {})
    
    to_delete = []
    for sid, info in strategies.items():
        sharpe_6m = info.get('sharpe_6m', 0)
        if sharpe_6m < threshold:
            to_delete.append(sid)
            
    logger.info(f"🔍 Found {len(to_delete)} strategies with Sharpe 6M < {threshold}")
    
    for sid in to_delete:
        factory.delete_strategy(sid)
        
    logger.info(f"✅ Successfully purged {len(to_delete)} strategies.")

if __name__ == "__main__":
    purge_low_quality_strategies(1.0)
