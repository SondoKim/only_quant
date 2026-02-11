import yaml
from pathlib import Path
from src.data.loader import DataLoader
from src.strategies.generator import StrategyGenerator

# Initialize components
loader = DataLoader()
generator = StrategyGenerator()

# Extract attributes needed for counting
assets = loader.all_tickers
related_assets = {}

# Build related assets map (identical logic to main.py)
cross_pairs = loader.get_cross_asset_pairs()
for category, pairs in cross_pairs.items():
    for pair in pairs:
        if len(pair) == 2:
            asset1, asset2 = pair
            if asset1 not in related_assets:
                related_assets[asset1] = []
            if asset2 not in related_assets[asset1]:
                related_assets[asset1].append(asset2)
            if asset2 not in related_assets:
                related_assets[asset2] = []
            if asset1 not in related_assets[asset2]:
                related_assets[asset2].append(asset1)

# Count strategies
counts = generator.count_strategies(assets, related_assets)

print("\n📈 Strategy Combination Summary:")
print(f"   Momentum: {counts['momentum']}")
print(f"   Mean Reversion: {counts['mean_reversion']}")
print(f"   Advanced: {counts['advanced']}")
print(f"   -------------------")
print(f"   TOTAL: {counts['total']}")
