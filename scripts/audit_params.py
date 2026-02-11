import json
import os
import yaml
from pathlib import Path

def audit_strategies(indicators_path, strategies_dir):
    # Load indicators config
    with open(indicators_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    tech_indicators = config.get('technical_indicators', {})
    cross_indicators = config.get('cross_asset_indicators', {})
    adv_strategies = config.get('advanced_strategies', {})
    
    # Combined indicators for easy lookup
    all_indicators = {**tech_indicators, **cross_indicators}
    
    mismatches = []
    
    # Iterate through strategies
    strategies_path = Path(strategies_dir)
    for file_path in strategies_path.glob('*.json'):
        if file_path.name == 'index.json':
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            strategy_name = strategy.get('strategy_name')
            params = strategy.get('params', {})
            
            is_outdated = False
            reasons = []
            
            # Check if it's a standard/cross indicator
            if strategy_name in all_indicators:
                indicator_config = all_indicators[strategy_name]
                config_params = indicator_config.get('params', {})
                config_thresholds = indicator_config.get('thresholds', {})
                
                # Check parameters
                for p_name, p_val in params.items():
                    if p_name in config_params:
                        allowed_vals = config_params[p_name]
                        if p_val not in allowed_vals:
                            is_outdated = True
                            reasons.append(f"Param '{p_name}': {p_val} not in {allowed_vals}")
                    elif p_name in config_thresholds:
                        allowed_vals = config_thresholds[p_name]
                        if p_val not in allowed_vals:
                            is_outdated = True
                            reasons.append(f"Threshold '{p_name}': {p_val} not in {allowed_vals}")
                    # Note: We don't necessarily mark as outdated if extra params exist unless strict
            
            # Check if it's an advanced strategy
            elif strategy_name in adv_strategies:
                adv_config = adv_strategies[strategy_name]
                for p_name, p_val in params.items():
                    if p_name in adv_config:
                        allowed_vals = adv_config[p_name]
                        # Some advanced params might be nested or lists
                        if isinstance(allowed_vals, list) and not isinstance(allowed_vals[0], list):
                            if p_val not in allowed_vals:
                                is_outdated = True
                                reasons.append(f"Adv Param '{p_name}': {p_val} not in {allowed_vals}")
            else:
                # strategy_name not found in indicators.yaml
                is_outdated = True
                reasons.append(f"Strategy name '{strategy_name}' not found in indicators.yaml")
                
            if is_outdated:
                mismatches.append({
                    'id': strategy.get('strategy_id'),
                    'name': strategy_name,
                    'reasons': reasons
                })
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return mismatches

if __name__ == "__main__":
    mismatches = audit_strategies(
        "config/indicators.yaml", 
        "src/factory/strategies"
    )
    
    print(f"Found {len(mismatches)} outdated strategies.\n")
    
    # Sort by name for better grouping
    mismatches.sort(key=lambda x: x['name'])
    
    for m in mismatches:
        print(f"[{m['id']}] {m['name']}")
        for r in m['reasons']:
            print(f"  - {r}")
        print()
