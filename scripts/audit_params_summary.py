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

def audit_strategies(indicators_path, strategies_dir):
    with open(indicators_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    tech_indicators = config.get('technical_indicators', {})
    cross_indicators = config.get('cross_asset_indicators', {})
    adv_strategies = config.get('advanced_strategies', {})
    
    all_configs = {**tech_indicators, **cross_indicators, **adv_strategies}
    
    outdated_vals = defaultdict(lambda: defaultdict(set))
    missing_strat = set()
    
    strategies_path = Path(strategies_dir)
    for file_path in strategies_path.glob('*.json'):
        if file_path.name == 'index.json': continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                strategy = json.load(f)
            
            s_id = strategy.get('strategy_id')
            s_name = strategy.get('strategy_name')
            params = strategy.get('params', {})
            
            config_key = STRATEGY_MAP.get(s_name)
            if not config_key or config_key not in all_configs:
                missing_strat.add(s_name)
                continue
            
            conf = all_configs[config_key]
            
            # For standard indicators, params are nested under 'params' and 'thresholds'
            if 'params' in conf or 'thresholds' in conf:
                allowed_p = conf.get('params', {})
                allowed_t = conf.get('thresholds', {})
                
                for p_name, p_val in params.items():
                    if p_name in allowed_p:
                        if p_val not in allowed_p[p_name]:
                            outdated_vals[s_name][p_name].add(p_val)
                    elif p_name in allowed_t:
                        if p_val not in allowed_t[p_name]:
                            outdated_vals[s_name][p_name].add(p_val)
            else:
                # For advanced strategies, params are at root level of the config
                for p_name, p_val in params.items():
                    if p_name in conf:
                        allowed = conf[p_name]
                        if isinstance(allowed, list) and not isinstance(allowed[0], list):
                            if p_val not in allowed:
                                outdated_vals[s_name][p_name].add(p_val)
                                
        except: continue
            
    return outdated_vals, missing_strat

if __name__ == "__main__":
    outdated, missing = audit_strategies("config/indicators.yaml", "src/factory/strategies")
    
    print("### Outdated Parameter Analysis\n")
    if outdated:
        for s_name, p_map in sorted(outdated.items()):
            print(f"[{s_name}]")
            for p_name, vals in sorted(p_map.items()):
                print(f"  - {p_name}: {sorted(list(vals))}")
            print()
    else:
        print("No outdated parameters found for matching strategies.")
        
    if missing:
        print("\n### Strategies with No Mapping in indicators.yaml")
        for m in sorted(list(missing)):
            print(f"  - {m}")
