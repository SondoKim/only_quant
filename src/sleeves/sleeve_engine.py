"""
Sleeve Engine — continuous, risk-targeted systematic macro positions.

Pipeline (all vectorized, causal — position[t] uses only data ≤ t):

  prices ──► directional returns (sign-normalized: long = duration / foreign-FX)
        ──► sleeve signals (Trend, Value, Carry), each continuous & z-scored
        ──► cross-sectional demean within asset class  (market-neutral RV)
        ──► combine sleeves (config weights) → per-asset conviction
        ──► inverse-vol sizing (equal risk per asset)
        ──► portfolio vol targeting (ex-ante trailing) → target positions

The output `target_positions` is a (dates × assets) DataFrame in DIRECTIONAL
space: +1 unit ≈ "target_asset_vol" of annualized risk, long duration / long
foreign currency. PnL = positions.shift(1) * directional_returns.

Design note on data: only futures/spot PRICES are available (no yields, no
external macro series). Trend and Value are computed from price alone. FX Carry
uses the rate-future differential from `carry_pairs` as a (defensible) proxy.
A proper rates carry/curve sleeve needs yield data and is intentionally omitted
rather than faked.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _zscore(df: pd.DataFrame, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
    """Rolling time-series z-score per column."""
    mp = min_periods or max(20, window // 3)
    mean = df.rolling(window, min_periods=mp).mean()
    std = df.rolling(window, min_periods=mp).std()
    return (df - mean) / std.replace(0.0, np.nan)


class SleeveEngine:
    """Build continuous, risk-targeted sleeve positions from a price panel."""

    def __init__(
        self,
        prices: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        assets_config_path: Optional[str] = None,
        yields: Optional[pd.DataFrame] = None,
    ):
        self.prices = prices.sort_index()
        self.cfg = config or {}

        # Signal-only yield panel (NOT traded). When provided, the Carry/Value
        # sleeves use real yields; otherwise they fall back to price proxies.
        self.yields = None
        if yields is not None and not yields.empty:
            self.yields = yields.reindex(self.prices.index).ffill()

        # ── Universes ────────────────────────────────────────────────────
        cols = list(self.prices.columns)
        self.rates_assets = [c for c in cols if 'Comdty' in c and 'NQ' not in c]
        self.fx_assets = [c for c in cols if 'Curncy' in c]

        # ── Sign map: normalize every series to "long = duration / long-foreign".
        #    Rates futures up = bond price up = long duration  → +1
        #    FX futures (EC1/BP1/AD1/JY1) up = foreign up      → +1
        #    KRW Curncy is USDKRW spot: up = KRW WEAKER         → -1
        self.sign = pd.Series(1.0, index=cols)
        for c in cols:
            if c == 'KRW Curncy':
                self.sign[c] = -1.0
        # allow config override
        for c, s in (self.cfg.get('sign_overrides', {}) or {}).items():
            if c in self.sign.index:
                self.sign[c] = float(s)

        # ── Carry pairs + yield maps from assets.yaml ─────────────────────
        self.carry_pairs = self._load_carry_pairs(assets_config_path)
        # tradeable_yield_map: futures ticker → its tenor govt-yield ticker
        # fx_short_yield:      fx ticker → its 2Y yield ticker (+ 'US' anchor)
        self.tradeable_yield_map, self.fx_short_yield = self._load_yield_maps(assets_config_path)

        # ── Parameters (with defaults) ───────────────────────────────────
        # Slower horizons (6-12m) are the robust TSMOM standard; the fast 63d
        # horizon whipsaws badly on rates — excluded by default.
        self.trend_horizons = self.cfg.get('trend_horizons', [126, 252])
        self.trend_z_window = self.cfg.get('trend_z_window', 252)
        self.value_window = self.cfg.get('value_window', 504)
        self.carry_window = self.cfg.get('carry_window', 252)
        self.signal_clip = self.cfg.get('signal_clip', 2.0)
        # FX carry construction: 'yield' = true 2Y carry differential (default;
        # economically correct, best in the recent 2022+ normal-rates regime);
        # 'price' = rate-differential momentum (stronger in the 2020-21 reflation
        # era). Within statistical noise full-sample. Rates carry is always
        # yield-based (and only active when yields are present).
        self.fx_carry_source = self.cfg.get('fx_carry_source', 'yield')

        self.sleeve_weights = self.cfg.get('sleeve_weights', {
            'trend': 1.0, 'value': 0.5, 'carry': 1.0,
        })
        # Per-sleeve cross-sectional neutralization (1.0 = fully market-neutral RV,
        # 0.0 = keep directional/time-series component). Trend's edge IS the
        # directional CTA premium, so it stays directional by default; Value and
        # Carry are relative-value and neutralized. Accepts a scalar for all.
        xn = self.cfg.get('xs_neutralize', {'trend': 0.0, 'value': 1.0, 'carry': 1.0})
        if isinstance(xn, (int, float)):
            xn = {'trend': float(xn), 'value': float(xn), 'carry': float(xn)}
        self.xs_neutralize = {k: float(v) for k, v in xn.items()}

        # Position smoothing (EWMA on conviction) to damp turnover. 0 = off.
        self.signal_smooth_span = int(self.cfg.get('signal_smooth_span', 5))

        # Regime gate (가): scale the TREND sleeve down in ranging regimes
        # (Hurst ≤ threshold) so it stops whipsawing in choppy markets.
        rg = self.cfg.get('regime_gate', {}) or {}
        self.regime_enabled = bool(rg.get('enabled', False))
        self.regime_method = rg.get('method', 'rs')
        self.regime_window = int(rg.get('window', 120))
        self.regime_threshold = float(rg.get('threshold', 0.5))
        self.regime_ranging_weight = float(rg.get('ranging_trend_weight', 0.0))
        self.regime_recompute_every = int(rg.get('recompute_every', 5))
        self._regime_gate_cache: Optional[pd.DataFrame] = None

        self.vol_halflife = self.cfg.get('vol_halflife', 33)      # ~ 60d window
        self.target_asset_vol = self.cfg.get('target_asset_vol', 0.10)
        self.target_port_vol = self.cfg.get('target_port_vol', 0.10)
        self.port_vol_window = self.cfg.get('port_vol_window', 63)
        self.max_asset_pos = self.cfg.get('max_asset_pos', 3.0)
        self.max_gross_leverage = self.cfg.get('max_gross_leverage', 10.0)

        # ── Derived series ───────────────────────────────────────────────
        # Directional returns: economically-meaningful return of a long position.
        self.dir_returns = self.prices.pct_change() * self.sign
        # Directional synthetic price index (for trend/value in directional space)
        self.dir_px = (1.0 + self.dir_returns.fillna(0.0)).cumprod()

    # ──────────────────────────────────────────────────────────────────────
    def _load_carry_pairs(self, assets_config_path: Optional[str]) -> Dict[str, Dict[str, str]]:
        path = assets_config_path or str(
            Path(__file__).parent.parent.parent / 'config' / 'assets.yaml'
        )
        try:
            with open(path, 'r', encoding='utf-8') as f:
                acfg = yaml.safe_load(f) or {}
        except Exception:
            return {}
        out = {}
        for pair in acfg.get('carry_pairs', []) or []:
            fx = pair.get('fx')
            if fx and fx in self.prices.columns:
                out[fx] = {'us_rate': pair.get('us_rate'), 'foreign_rate': pair.get('foreign_rate')}
        return out

    def _load_yield_maps(self, assets_config_path: Optional[str]):
        """Load signal-yield maps from assets.yaml → (tradeable_yield_map, fx_short_yield)."""
        path = assets_config_path or str(
            Path(__file__).parent.parent.parent / 'config' / 'assets.yaml'
        )
        try:
            with open(path, 'r', encoding='utf-8') as f:
                acfg = yaml.safe_load(f) or {}
        except Exception:
            return {}, {}
        sy = acfg.get('signal_yields', {}) or {}
        return (sy.get('tradeable_yield_map', {}) or {},
                sy.get('fx_short_yield', {}) or {})

    def _y(self, ticker: Optional[str]) -> Optional[pd.Series]:
        """Fetch a yield series by ticker, or None if unavailable."""
        if self.yields is None or ticker is None or ticker not in self.yields.columns:
            return None
        s = self.yields[ticker]
        return s if s.notna().any() else None

    def _has_fx_yields(self) -> bool:
        us = self.fx_short_yield.get('US')
        return self._y(us) is not None and any(
            self._y(self.fx_short_yield.get(fx)) is not None for fx in self.fx_assets
        )

    def _has_rates_yields(self) -> bool:
        return any(self._y(self.tradeable_yield_map.get(a)) is not None
                   for a in self.rates_assets)

    # ──────────────────────────────────────────────────────────────────────
    # Sleeve signals (continuous, ~z-score scaled, directional space)
    # ──────────────────────────────────────────────────────────────────────
    def trend_signal(self, assets: List[str]) -> pd.DataFrame:
        """Multi-horizon, vol-normalized time-series momentum."""
        px = self.dir_px[assets]
        parts = []
        for h in self.trend_horizons:
            ret_h = px.pct_change(h)
            parts.append(_zscore(ret_h, self.trend_z_window))
        sig = pd.concat(parts).groupby(level=0).mean()
        return sig.clip(-self.signal_clip, self.signal_clip)

    def value_signal(self, assets: List[str], asset_class: str = 'rates') -> pd.DataFrame:
        """Long-horizon mean reversion: expensive (high vs 2y mean) → short.

        Price-based for ALL classes. (A yield-level version was tested for rates
        but performed worse — the continuous future embeds roll-down that the
        cash yield does not, and the price reversal captures it better.)
        """
        px = self.dir_px[assets]
        z = _zscore(np.log(px.clip(lower=1e-9)), self.value_window)
        return (-z).clip(-self.signal_clip, self.signal_clip)

    def carry_signal(self, assets: List[str], asset_class: str) -> pd.DataFrame:
        """Carry sleeve, yield-based when data is present.

        FX: differential (foreign 2Y − US 2Y). Higher → long foreign currency.
            Fallback (no yields): price proxy (us_rate_price − foreign_rate_price).
        Rates: yield LEVEL per market (cross-country bond carry). Higher yield
            market → LONG (xs-demean later makes it long high / short low).
            No price fallback (returns empty → sleeve simply inactive on rates).
        """
        if asset_class == 'fx':
            # fx_carry_source: 'price' (rate-differential momentum, empirically
            # stronger 2020-26) or 'yield' (true carry = foreign 2Y − US 2Y).
            use_yield = (self.fx_carry_source == 'yield') and self._has_fx_yields()
            if use_yield:
                us = self._y(self.fx_short_yield.get('US'))
                diffs = {}
                for fx in assets:
                    fy = self._y(self.fx_short_yield.get(fx))
                    if fy is not None and us is not None:
                        diffs[fx] = fy - us                      # yield-point differential
                if diffs:
                    z = _zscore(pd.DataFrame(diffs), self.carry_window)
                    return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)
            # price-proxy (rate-differential momentum) — default
            diffs = {}
            for fx, rp in self.carry_pairs.items():
                if fx not in assets:
                    continue
                ur, fr = rp.get('us_rate'), rp.get('foreign_rate')
                if ur in self.prices.columns and fr in self.prices.columns:
                    diffs[fx] = self.prices[ur] - self.prices[fr]
            if not diffs:
                return pd.DataFrame(index=self.prices.index, columns=assets)
            z = _zscore(pd.DataFrame(diffs), self.carry_window)
            return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)

        # rates carry — yield level, only when yields available
        if asset_class == 'rates' and self._has_rates_yields():
            ydf = {}
            for a in assets:
                ys = self._y(self.tradeable_yield_map.get(a))
                if ys is not None:
                    ydf[a] = ys
            if ydf:
                z = _zscore(pd.DataFrame(ydf), self.carry_window)
                return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)
        return pd.DataFrame(index=self.prices.index, columns=assets)

    # ──────────────────────────────────────────────────────────────────────
    def _regime_gate(self, assets: List[str]) -> pd.DataFrame:
        """(가) Per-asset trend multiplier from rolling Hurst.

        H > threshold (trending) → 1.0 (full trend).
        H ≤ threshold (ranging)  → ranging_trend_weight (e.g. 0 = trend off).
        Hurst is recomputed every `recompute_every` days and forward-filled
        (regimes drift slowly), then lightly EWMA-smoothed to avoid flip-flop.
        Causal: window ends at t, applied to t's signal (positions lag by 1).
        """
        from ..indicators.technical import TechnicalIndicators
        if self._regime_gate_cache is None:
            self._regime_gate_cache = {}   # per-asset cache of gate Series
        cache = self._regime_gate_cache

        idx = self.prices.index
        step = max(1, self.regime_recompute_every)
        sample_points = idx[self.regime_window::step]

        for asset in assets:
            if asset in cache:
                continue
            s = self.dir_px[asset]
            h_vals = {}
            for t in sample_points:
                window_slice = s.loc[:t].iloc[-self.regime_window:]
                h_vals[t] = TechnicalIndicators.hurst(
                    window_slice, window=self.regime_window, method=self.regime_method
                )
            if not h_vals:
                cache[asset] = pd.Series(1.0, index=idx)
                continue
            h_series = pd.Series(h_vals).reindex(idx).ffill()
            g = np.where(h_series > self.regime_threshold, 1.0, self.regime_ranging_weight)
            cache[asset] = pd.Series(g, index=idx).ewm(span=step * 2, min_periods=1).mean().fillna(1.0)

        return pd.DataFrame({a: cache[a] for a in assets})

    @staticmethod
    def _xs_demean(sig: pd.DataFrame, strength: float) -> pd.DataFrame:
        """Cross-sectionally demean each row by `strength` (market-neutral RV)."""
        if sig.empty or strength <= 0:
            return sig
        return sig - strength * sig.mean(axis=1).values.reshape(-1, 1)

    def _combine_class(self, assets: List[str], asset_class: str) -> pd.DataFrame:
        """Weighted conviction for one asset class.

        Each sleeve is neutralized by ITS OWN xs_neutralize strength before
        weighting: Trend stays directional (time-series), Value/Carry become
        relative-value (market-neutral within the class). Carry is active on FX
        always (yield or price proxy) and on rates only when yields are present.
        """
        w = self.sleeve_weights
        combined = pd.DataFrame(0.0, index=self.prices.index, columns=assets)
        contrib = 0.0

        trend = self.trend_signal(assets)
        if self.regime_enabled:
            # Gate trend down in ranging regimes BEFORE neutralization
            trend = trend * self._regime_gate(assets)
        sleeves = {'trend': trend,
                   'value': self.value_signal(assets, asset_class)}

        include_carry = (asset_class == 'fx') or \
                        (asset_class == 'rates' and self._has_rates_yields())
        if include_carry:
            sleeves['carry'] = self.carry_signal(assets, asset_class).reindex(columns=assets)

        for name, sig in sleeves.items():
            wt = w.get(name, 0.0)
            if wt == 0.0 or sig is None or sig.empty:
                continue
            sig = self._xs_demean(sig.fillna(0.0), self.xs_neutralize.get(name, 0.0))
            combined = combined.add(wt * sig, fill_value=0.0)
            contrib += abs(wt)

        if contrib > 0:
            combined = combined / contrib
        if self.signal_smooth_span > 1:
            combined = combined.ewm(span=self.signal_smooth_span, min_periods=1).mean()
        return combined

    # ──────────────────────────────────────────────────────────────────────
    def _realized_vol(self) -> pd.DataFrame:
        """EWMA annualized realized vol per asset (directional returns)."""
        vol = self.dir_returns.ewm(halflife=self.vol_halflife, min_periods=20).std()
        return vol * np.sqrt(TRADING_DAYS)

    def compute_target_positions(self) -> pd.DataFrame:
        """Full pipeline → (dates × assets) directional target positions."""
        conviction = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)

        if self.rates_assets:
            rates_conv = self._combine_class(self.rates_assets, 'rates')
            conviction[self.rates_assets] = rates_conv
        if self.fx_assets:
            fx_conv = self._combine_class(self.fx_assets, 'fx')
            conviction[self.fx_assets] = fx_conv

        traded = self.rates_assets + self.fx_assets
        conviction = conviction[traded].fillna(0.0)

        # ── Inverse-vol sizing: equal risk per asset ─────────────────────
        vol = self._realized_vol()[traded]
        inv_vol = (self.target_asset_vol / vol).clip(upper=self.max_asset_pos * 5)
        pos = (conviction * inv_vol).clip(-self.max_asset_pos, self.max_asset_pos)
        pos = pos.fillna(0.0)

        # ── Portfolio vol targeting (ex-ante, trailing) ──────────────────
        pos = self._apply_portfolio_vol_target(pos, traded)

        return pos

    def _apply_portfolio_vol_target(self, pos: pd.DataFrame, traded: List[str]) -> pd.DataFrame:
        """Scale whole book each day so trailing portfolio vol ≈ target.

        Ex-ante: today's weights applied to PAST returns (no look-ahead).
        """
        rets = self.dir_returns[traded].fillna(0.0)
        win = self.port_vol_window
        scales = pd.Series(1.0, index=pos.index)

        pos_vals = pos.values
        ret_vals = rets.values
        n = len(pos)
        for i in range(n):
            if i < win:
                scales.iloc[i] = 0.0  # warm-up: stay flat
                continue
            w = pos_vals[i]
            if not np.any(w):
                continue
            window_rets = ret_vals[i - win:i]          # past returns only
            port_hist = window_rets @ w                 # portfolio return path
            pv = port_hist.std() * np.sqrt(TRADING_DAYS)
            if pv > 1e-9:
                scales.iloc[i] = min(self.target_port_vol / pv,
                                     self.max_gross_leverage / (np.abs(w).sum() + 1e-9))
        scaled = pos.mul(scales, axis=0)
        # final gross leverage cap
        gross = scaled.abs().sum(axis=1)
        over = gross > self.max_gross_leverage
        if over.any():
            scaled.loc[over] = scaled.loc[over].mul(
                self.max_gross_leverage / gross[over], axis=0
            )
        return scaled.fillna(0.0)
