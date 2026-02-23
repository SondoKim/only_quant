"""
Indicator Cache for Fast Strategy Discovery

Caches computed indicator values to avoid redundant calculations
when multiple strategies use the same indicator on the same asset.
"""

from typing import Dict, Any, Tuple
import pandas as pd


class IndicatorCache:
    """Cache computed indicators keyed by (asset, indicator_name, params)."""
    
    def __init__(self):
        self._cache: Dict[Tuple, Any] = {}
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, asset: str, indicator_name: str, params: dict) -> tuple:
        """Create hashable cache key."""
        param_tuple = tuple(sorted(params.items())) if params else ()
        return (asset, indicator_name, param_tuple)
    
    def get(self, asset: str, indicator_name: str, params: dict):
        """Get cached result or None."""
        key = self._make_key(asset, indicator_name, params)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result
    
    def put(self, asset: str, indicator_name: str, params: dict, value):
        """Store result in cache."""
        key = self._make_key(asset, indicator_name, params)
        self._cache[key] = value
    
    def get_or_compute(self, asset: str, indicator_name: str, params: dict, compute_fn):
        """Get from cache or compute and store."""
        result = self.get(asset, indicator_name, params)
        if result is not None:
            return result
        result = compute_fn()
        self.put(asset, indicator_name, params, result)
        return result
    
    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': f"{self._hits / total * 100:.1f}%" if total > 0 else "N/A",
            'cached_items': len(self._cache),
        }
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
