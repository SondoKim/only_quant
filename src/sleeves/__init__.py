"""Sleeve-based systematic macro engine.

A clean parallel to the strategy-factory: instead of mining thousands of
binary technical strategies, it builds a small set of economically-grounded
risk-premium sleeves (Trend / Value / Carry) as CONTINUOUS, cross-sectionally
market-neutral signals, then risk-sizes them to a portfolio volatility target.
"""

from .sleeve_engine import SleeveEngine

__all__ = ["SleeveEngine"]
