import json
import os
import yaml
from pathlib import Path
from collections import defaultdict

# Mapping from strategy_name in JSON to config key in indicators.yaml
STRATEGY_MAP = {
    'ma_crossover': 'sma_crossover',
    'ema_crossover': 'ema_crossover',
    'rsi_momentum': 'rsi',
    'macd_crossover': 'macd',
    'rate_of_change': 'rate_of_change',
    'spread_momentum': 'spread_zscore',
    'zscore_reversion': 'zscore',
    'rsi_extremes': 'rsi',
    'bollinger_reversion': 'bollinger_bands',
    'spread_zscore_reversion': 'spread_zscore',
    'spread_percentile_reversion': 'spread_percentile',
    'filtered_momentum': 'filtered_momentum',
    'lead_lag_momentum': 'lead_lag_momentum',
    'multi_tf_momentum': 'multi_tf_momentum',
    'volatility_breakout': 'volatility_breakout',
    'relative_strength_rank': 'relative_strength_rank'
}

def identify_unused_parameters(indicators_path, strategies_dir):
    with open(indicators_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    tech_indicators = config.get('technical_indicators', {})
    cross_indicators = config.get('cross_asset_indicators', {})
    adv_strategies = config.get('advanced_strategies', {})
    
    all_configs = {**tech_indicators, **cross_indicators, **adv_strategies}
    
    # Track used values for each config key and its parameters
    # used_values[config_key][param_name] = set(values)
    used_values = defaultdict(lambda: defaultdict(set))
    
    strategies_path = Path(strategies_dir)
    for file_path in strategies_path.glob('*.json'):
        if file_path.name == 'index.json': continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            s_name = strategy.get('strategy_name')
            params = strategy.get('params', {})
            
            config_key = STRATEGY_MAP.get(s_name)
            if config_key and config_key in all_configs:
                for p_name, p_val in params.items():
                    used_values[config_key][p_name].add(p_val)
        except: continue
    
    unused_report = defaultdict(lambda: defaultdict(list))
    
    for config_key, conf in all_configs.items():
        # Handle standard indicators (nested params/thresholds)
        if 'params' in conf or 'thresholds' in conf:
            # Check 'params'
            for p_name, possible_vals in conf.get('params', {}).items():
                used = used_values[config_key].get(p_name, set())
                unused = [v for v in possible_vals if v not in used]
                if unused:
                    unused_report[config_key][p_name] = unused
            
            # Check 'thresholds'
            for p_name, possible_vals in conf.get('thresholds', {}).items():
                used = used_values[config_key].get(p_name, set())
                unused = [v for v in possible_vals if v not in used]
                if unused:
                    unused_report[config_key][p_name] = unused
        else:
            # Handle advanced strategies (root level params)
            for p_name, possible_vals in conf.items():
                if isinstance(possible_vals, list) and not isinstance(possible_vals[0], list):
                    used = used_values[config_key].get(p_name, set())
                    unused = [v for v in possible_vals if v not in used]
                    if unused:
                        unused_report[config_key][p_name] = unused
                        
    return unused_report

if __name__ == "__main__":
    report = identify_unused_parameters("config/indicators.yaml", "src/factory/strategies")
    
    print("### Unused Parameter Values (In indicators.yaml but NOT in factory)\n")
    if report:
        for config_key, params in sorted(report.items()):
            print(f"[{config_key}]")
            for p_name, unused_vals in sorted(params.items()):
                print(f"  - {p_name}: {unused_vals}")
            print()
    else:
        print("All parameter values from indicators.yaml are currently used in the factory.")
