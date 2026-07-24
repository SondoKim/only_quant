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
        # exclude_assets: 거래 유니버스에서만 제외 (가격 컬럼은 유지 — FX 캐리
        # price-proxy 등이 금리선물 가격을 참조할 수 있으므로 데이터는 남긴다).
        # 2026-07-14 유니버스 축소 A/B (scripts/test_universe_reduction.py):
        # 유로존 4종 제외 시 rates SR 0.88→1.20, MaxDD -8.8%→-5.0%, H1/H2 모두 개선.
        cols = list(self.prices.columns)
        excluded = set(self.cfg.get('exclude_assets', []) or [])
        self.rates_assets = [c for c in cols
                             if 'Comdty' in c and 'NQ' not in c and c not in excluded]
        self.fx_assets = [c for c in cols if 'Curncy' in c and c not in excluded]

        # signal_only_assets: 시그널 유니버스에는 그대로 두고 '매매'에서만 뺀다.
        # exclude_assets 와의 차이 — exclude 는 자산을 횡단면에서 통째로 지우므로
        # xs-demean 기준선(글로벌 듀레이션 평균)과 리버전 페이드가 전부 바뀐다.
        # signal_only 는 z-score·demean·리버전 계산에는 계속 참여시키고 최종
        # 포지션만 0으로 눌러, 남은 자산이 볼타겟팅으로 리스크를 흡수하게 한다.
        # (호주처럼 시그널 정보력은 있으나 집행이 부담스러운 시장용.)
        self.signal_only_assets = set(self.cfg.get('signal_only_assets', []) or [])

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
        # curve_slope_map:     long-end futures → [short yield, long yield]
        # policy_rate_map:     futures → country 2Y/3Y (policy-proxy) yield
        (self.tradeable_yield_map, self.fx_short_yield,
         self.curve_slope_map, self.policy_rate_map) = self._load_yield_maps(assets_config_path)

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
            # Rates-only directional sleeves (need curve_slope_map / policy_rate_map
            # yields; silently inactive without them). Default 0 = off until A/B'd.
            'curve': 0.0, 'policy': 0.0,
        })
        # Policy momentum lookback (2Y yield change, trading days)
        self.policy_period = int(self.cfg.get('policy_period', 126))
        # Per-sleeve cross-sectional neutralization (1.0 = fully market-neutral RV,
        # 0.0 = keep directional/time-series component). Trend's edge IS the
        # directional CTA premium, so it stays directional by default; Value and
        # Carry are relative-value and neutralized. Accepts a scalar for all.
        xn = self.cfg.get('xs_neutralize', {'trend': 0.0, 'value': 1.0, 'carry': 1.0,
                                            'curve': 0.0, 'policy': 0.0})
        if isinstance(xn, (int, float)):
            xn = {'trend': float(xn), 'value': float(xn), 'carry': float(xn)}
        self.xs_neutralize = {k: float(v) for k, v in xn.items()}
        # FX 북 전용 오버라이드 (없으면 xs_neutralize 공유 = 기존 동작).
        # 금리 쪽 중립화 실험이 FX 북을 건드리지 않게 분리하는 용도.
        xnf = self.cfg.get('xs_neutralize_fx')
        self.xs_neutralize_fx = ({k: float(v) for k, v in xnf.items()}
                                 if isinstance(xnf, dict) else None)

        # Position smoothing (EWMA on conviction) to damp turnover. 0 = off.
        self.signal_smooth_span = int(self.cfg.get('signal_smooth_span', 5))
        # EWMA on FINAL positions (fraction of prior held; 0 = off).
        self.position_smooth = float(self.cfg.get('position_smooth', 0.0) or 0.0)

        # Book stop (2026-06-11): shadow-equity trailing stop on the RATES
        # book. Watches the hypothetical (un-stopped) book PnL; drawdown
        # beyond dd_half → halve rates positions, beyond dd_flat → flat,
        # automatically re-engages as the shadow book recovers. Responds to
        # REALIZED damage instead of predicting inflections — fast price/vol
        # regime gates (21d veto, 63d agreement, vol brake) all LOWERED SR in
        # the 2016-26 A/B; this kept SR (~0.5) while cutting MaxDD -33%→-14%.
        #
        # ⚠ 2026-07-22 감사에서 드러난 원설계의 결함 — dd 를 산술 cumsum 의
        # all-time cummax 대비로 재던 탓에, 섀도우 북이 옛 고점을 못 넘으면
        # dd 가 영구히 임계 위에 고착돼 스톱이 킬스위치가 됐다 (마지막 신고점
        # 2022-09-28 → 2023-07 이후 계속 플랫, 2016+ 42%일 플랫). 두 가지를
        # 바꿨다: ① 자산곡선을 복리(cumprod)로, dd 를 고점 대비 '비율'로 —
        # 북 규모가 변해도 임계의 의미가 일정하다. ② 고점 기준을 all-time 이
        # 아니라 dd_window 일 롤링 최대로 — 오래된 고점은 시간이 지나면
        # 굴러떨어지므로, 북이 횡보만 해도 재진입이 열린다 (0 = 옛 all-time).
        bs = self.cfg.get('book_stop', {}) or {}
        self.book_stop_enabled = bool(bs.get('enabled', False))
        self.book_stop_dd_half = float(bs.get('dd_half', 4.0))   # 고점대비 %
        self.book_stop_dd_flat = float(bs.get('dd_flat', 8.0))
        self.book_stop_dd_window = int(bs.get('dd_window', 0) or 0)

        # Reversion sub-book (2026-06-11, prop-desk "don't sit flat" request):
        # fast CROSS-SECTIONAL reversal on rates — fade each market's 10d move
        # relative to the global-duration average (market-neutral, so it does
        # not fight trends; ts-fade of the aggregate move was tested and lost
        # -0.5 SR). Always-on (Hurst range-gating LOWERED SR 0.70→0.51), runs
        # OUTSIDE the book stop, blended with the main book in
        # finalize_positions. ⚠ COST CLIFF: net SR 0.70 @0.5bp, 0.31 @1bp,
        # ≤0 @1.5bp — requires passive execution.
        rv = self.cfg.get('reversion', {}) or {}
        self.reversion_enabled = bool(rv.get('enabled', False))
        self.reversion_weight = float(rv.get('weight', 0.5))
        self.reversion_lookback = int(rv.get('lookback', 10))
        self.reversion_z_window = int(rv.get('z_window', 120))
        self.reversion_smooth = float(rv.get('smooth', 0.5))

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

        # ── Curve trades (2026-07-23): DV01(≈vol)-중립 스티프너를 합성 자산으로
        # 추가해 실효 독립 베팅 수를 늘린다 (현행 북은 사실상 미국·한국 듀레이션
        # 2개). 합성 수익률 = H·r_short − r_long, H = (σ_long/σ_short) 전일값
        # (causal). 시그널·사이징·볼타겟·북스톱은 합성 공간에서 돌고, 최종 출력
        # 직전에 실계약 레그(short-end +H, long-end −1)로 환산되므로 하위 소비자
        # (백테스트·대시보드·라이브 피드)는 실자산 컬럼만 본다. 페어는 양 레그가
        # 모두 '매매 가능'(exclude/signal_only 아님)일 때만 생성 — 커브는 동일
        # 국가 동시 마감이라 비동시 종가 아티팩트가 구조적으로 없다.
        ct = self.cfg.get('curve_trades', {}) or {}
        self.curve_enabled = bool(ct.get('enabled', False))
        self.curve_pairs_cfg = ct.get('pairs') or [
            {'name': 'US2s10s', 'short_leg': 'TU1 Comdty', 'long_leg': 'TY1 Comdty'},
            {'name': 'KR3s10s', 'short_leg': 'KE1 Comdty', 'long_leg': 'KAA1 Comdty'},
        ]
        # 커브 슬리브 가중 — 본북의 '전 슬리브 균등' 전례를 따라 사전등록 균등.
        # carry = 기울기 리치니스의 음수 (가파름 → 플래트너 롤다운 유리 → 숏
        # 스티프너), policy = short-leg 국가 2Y 변화 (완화 → 불 스티프닝 → 롱).
        self.curve_sleeve_weights = ct.get('sleeve_weights', {
            'trend': 1.0, 'value': 1.0, 'carry': 1.0, 'policy': 1.0})
        self.curve_risk_weight = float(ct.get('risk_weight', 1.0))
        self.curve_hedge_clip = float(ct.get('hedge_clip', 10.0))

        # ── Block risk budgeting (2026-07-23): 인버스볼 equal-risk-per-asset 은
        # ρ(TU,TY)=0.78, ρ(KE,KAA)=0.87 인 국가 블록에 예산을 이중 배분한다.
        # 블록 내 평균 상관(롤링, causal)으로 블록 합산 변동성을 추정해 각 블록이
        # '한 개 베팅'만큼의 리스크를 갖도록 멤버를 축소 — 이후 포트폴리오 볼타겟이
        # 전체를 재확대하므로 순효과는 블록 간 상대 예산 재배분이다.
        rb = self.cfg.get('risk_blocks', {}) or {}
        self.risk_blocks_enabled = bool(rb.get('enabled', False))
        self.risk_blocks_corr_window = int(rb.get('corr_window', 252))
        self.risk_blocks_cfg = rb.get('blocks') or {
            'US_dur': ['TU1 Comdty', 'TY1 Comdty'],
            'KR_dur': ['KE1 Comdty', 'KAA1 Comdty'],
            'AU_dur': ['YM1 Comdty', 'XM1 Comdty'],
        }

        self.vol_halflife = self.cfg.get('vol_halflife', 33)      # ~ 60d window
        self.target_asset_vol = self.cfg.get('target_asset_vol', 0.10)
        self.target_port_vol = self.cfg.get('target_port_vol', 0.10)
        self.port_vol_window = self.cfg.get('port_vol_window', 63)
        self.max_asset_pos = self.cfg.get('max_asset_pos', 3.0)
        self.max_gross_leverage = self.cfg.get('max_gross_leverage', 10.0)
        # 금리 북 최종 노출 스케일 (2026-07-16 운용 캘리브레이션: 운용 북 대비
        # 델타 과대 → 0.5). finalize_positions 맨 끝에서 금리 자산에만 곱한다.
        self.rates_exposure_scale = float(self.cfg.get('rates_exposure_scale', 1.0))

        # ── Derived series ───────────────────────────────────────────────
        # Directional returns: economically-meaningful return of a long position.
        self.dir_returns = self.prices.pct_change() * self.sign
        # Directional synthetic price index (for trend/value in directional space)
        self.dir_px = (1.0 + self.dir_returns.fillna(0.0)).cumprod()

        # Curve synthetics — dir_returns/dir_px 에만 컬럼을 추가한다 (prices 에는
        # 넣지 않음: 'Comdty' 스캔 기반 유니버스 감지를 오염시키지 않기 위해).
        self.curve_assets: List[str] = []
        self._curve_legs: Dict[str, Any] = {}
        if self.curve_enabled:
            self._build_curve_assets()

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
            return {}, {}, {}, {}
        sy = acfg.get('signal_yields', {}) or {}
        return (sy.get('tradeable_yield_map', {}) or {},
                sy.get('fx_short_yield', {}) or {},
                sy.get('curve_slope_map', {}) or {},
                sy.get('policy_rate_map', {}) or {})

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
        Rates: USD-HEDGED yield level per market (cross-country bond carry from
            a USD investor's seat). Hedge cost is proxied by the short-rate
            differential (covered interest parity):
                hedged_yield = local_yield − (local_2Y − US_2Y)
            so a high local yield financed by an equally high local short rate
            (e.g. classic EM/JPY cases in reverse) no longer scores as free
            carry. US assets: hedge cost = 0 → raw yield. Higher hedged yield
            → LONG (xs-demean later makes it long high / short low).
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

        # rates carry — USD-hedged yield level, only when yields available.
        # hedged = local_yield − (local funding(2Y/3Y) − US 2Y)  [CIP 근사]
        if asset_class == 'rates' and self._has_rates_yields():
            us_fund = self._y(self.fx_short_yield.get('US'))   # USD 펀딩 앵커 (US 2Y)
            ydf = {}
            for a in assets:
                ys = self._y(self.tradeable_yield_map.get(a))
                if ys is None:
                    continue
                fund = self._y(self.policy_rate_map.get(a))
                if us_fund is not None and fund is not None:
                    ydf[a] = ys - (fund - us_fund)     # 환헤지 후 달러환산 금리
                else:
                    ydf[a] = ys                        # 펀딩 데이터 없으면 원금리 폴백
            if ydf:
                z = _zscore(pd.DataFrame(ydf), self.carry_window)
                return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)
        return pd.DataFrame(index=self.prices.index, columns=assets)

    def curve_signal(self, assets: List[str]) -> pd.DataFrame:
        """Own-curve slope carry (rates, time-series/directional).

        slope = long-end yield − funding (2Y) yield from curve_slope_map. A
        steep curve = positive carry + rolldown for holding duration → LONG;
        inverted → SHORT. Z-scored vs its own trailing history so the signal
        is the slope's RICHNESS, not its absolute level. Empty when the curve
        yields are unavailable (sleeve silently inactive).
        """
        cols = {}
        for a in assets:
            pair = self.curve_slope_map.get(a)
            if not pair or len(pair) != 2:
                continue
            y_s, y_l = self._y(pair[0]), self._y(pair[1])
            if y_s is not None and y_l is not None:
                cols[a] = y_l - y_s
        if not cols:
            return pd.DataFrame(index=self.prices.index, columns=assets)
        z = _zscore(pd.DataFrame(cols), self.carry_window)
        return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)

    def policy_signal(self, assets: List[str]) -> pd.DataFrame:
        """Central-bank cycle momentum (rates, time-series/directional).

        Change of the country's 2Y/3Y (policy-proxy) yield over policy_period
        days: falling short yield = easing cycle → LONG duration; rising =
        hiking → SHORT. Continuous version of the factory's policy_momentum
        (its strongest alpha4). Empty without policy_rate_map yields.
        """
        cols = {}
        for a in assets:
            y = self._y(self.policy_rate_map.get(a))
            if y is not None:
                cols[a] = -(y - y.shift(self.policy_period))
        if not cols:
            return pd.DataFrame(index=self.prices.index, columns=assets)
        z = _zscore(pd.DataFrame(cols), self.trend_z_window)
        return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)

    # ──────────────────────────────────────────────────────────────────────
    # Curve trades (synthetic steepeners)
    # ──────────────────────────────────────────────────────────────────────
    def _build_curve_assets(self) -> None:
        """합성 스티프너 수익률 시리즈를 dir_returns/dir_px 에 추가.

        steepener return(t) = H(t)·r_short(t) − r_long(t),
        H(t) = (σ_long/σ_short) 의 전일(shift 1) EWMA 값 — 헤지는 t−1 에
        설정되므로 t−1 까지의 정보만 사용 (causal). 단위 포지션 1 = short-end
        선물 +H 계약분 / long-end 선물 −1 계약분 (레그 환산은
        _explode_curve_positions 에서 동일한 H 로 수행 → PnL 항등).
        """
        vol = self.dir_returns.ewm(halflife=self.vol_halflife, min_periods=20).std()
        for p in self.curve_pairs_cfg:
            s, l = p.get('short_leg'), p.get('long_leg')
            name = p.get('name') or f'{s}|{l} curve'
            if s not in self.rates_assets or l not in self.rates_assets:
                continue
            if s in self.signal_only_assets or l in self.signal_only_assets:
                continue   # 커브는 양 레그를 실제로 매매할 수 있어야만 성립
            H = (vol[l] / vol[s]).clip(lower=1.0 / self.curve_hedge_clip,
                                       upper=self.curve_hedge_clip).shift(1)
            r = (H * self.dir_returns[s] - self.dir_returns[l]).fillna(0.0)
            self.dir_returns[name] = r
            self.dir_px[name] = (1.0 + r).cumprod()
            self._curve_legs[name] = (s, l, H)
            self.curve_assets.append(name)
        if self.curve_assets:
            logger.info("Curve trades active: %s", ', '.join(self.curve_assets))

    def _combine_curve(self) -> pd.DataFrame:
        """커브 합성 자산의 결합 컨빅션 (전부 시계열 — xs-demean 없음).

        trend/value 는 합성 가격지수에 본북과 동일한 계산을 적용.
        carry  = −z(기울기): 가파른 커브 → 플래트너 롤다운 유리 → 숏 스티프너.
        policy = z(−Δ short-leg 정책금리): 완화 사이클 → 불 스티프닝 → 롱.
        """
        A = self.curve_assets
        w = self.curve_sleeve_weights
        sleeves: Dict[str, pd.DataFrame] = {}
        if w.get('trend', 0.0):
            sleeves['trend'] = self.trend_signal(A)
        if w.get('value', 0.0):
            sleeves['value'] = self.value_signal(A, 'curve')
        if w.get('carry', 0.0):
            cols = {}
            for name, (s, l, _h) in self._curve_legs.items():
                pair = self.curve_slope_map.get(l)
                if not pair or len(pair) != 2:
                    continue
                y_s, y_l = self._y(pair[0]), self._y(pair[1])
                if y_s is not None and y_l is not None:
                    cols[name] = -(y_l - y_s)
            if cols:
                z = _zscore(pd.DataFrame(cols), self.carry_window)
                sleeves['carry'] = z.reindex(columns=A).clip(
                    -self.signal_clip, self.signal_clip)
        if w.get('policy', 0.0):
            cols = {}
            for name, (s, l, _h) in self._curve_legs.items():
                y = self._y(self.policy_rate_map.get(s))
                if y is not None:
                    cols[name] = -(y - y.shift(self.policy_period))
            if cols:
                z = _zscore(pd.DataFrame(cols), self.trend_z_window)
                sleeves['policy'] = z.reindex(columns=A).clip(
                    -self.signal_clip, self.signal_clip)

        combined = pd.DataFrame(0.0, index=self.prices.index, columns=A)
        contrib = 0.0
        for nm, sig in sleeves.items():
            wt = w.get(nm, 0.0)
            if wt == 0.0 or sig is None or sig.empty:
                continue
            combined = combined.add(wt * sig.fillna(0.0), fill_value=0.0)
            contrib += abs(wt)
        if contrib > 0:
            combined = combined / contrib
        if self.signal_smooth_span > 1:
            combined = combined.ewm(span=self.signal_smooth_span, min_periods=1).mean()
        return combined * self.curve_risk_weight

    def _explode_curve_positions(self, pos: pd.DataFrame) -> pd.DataFrame:
        """합성 커브 포지션 → 실계약 레그 환산 (아웃라이트 포지션에 합산).

        단위 스티프너 p → short-end 레그 +p·H, long-end 레그 −p. 합성 수익률
        정의와 같은 H 를 쓰므로 레그 공간 PnL ≈ 합성 공간 PnL (1일 헤지비율
        드리프트만 차이 — EWMA halflife 33d 라 무시 가능). 비용은 레그 회전에서
        자연히 두 다리 몫으로 계산된다.
        """
        pos = pos.copy()
        for name, (s, l, H) in self._curve_legs.items():
            if name not in pos.columns:
                continue
            p = pos[name]
            h = H.reindex(pos.index)
            pos[s] = pos[s].add((p * h).fillna(0.0), fill_value=0.0)
            pos[l] = pos[l].sub(p.fillna(0.0), fill_value=0.0)
            pos = pos.drop(columns=[name])
        return pos

    # ──────────────────────────────────────────────────────────────────────
    def _apply_risk_blocks(self, pos: pd.DataFrame, traded: List[str]) -> pd.DataFrame:
        """블록(국가 듀레이션) 단위 리스크 예산 정렬.

        블록 멤버 m 개가 각각 unit-risk 면 블록 변동성 ≈ √(m·(1+(m−1)·ρ̄)) unit.
        멤버를 그 역수로 축소해 모든 블록이 싱글톤과 같은 '한 개 베팅' 리스크를
        갖게 한다. ρ̄ = 블록 내 평균 페어와이즈 롤링 상관 (shift 1 → causal;
        워밍업 구간은 ρ̄=1 가정 = 1/m, 보수적). 미등록 자산·커브 합성은 암묵적
        싱글톤(스케일 1). 이후 포트폴리오 볼타겟이 전체 레벨을 복원한다.
        """
        if not self.risk_blocks_enabled:
            return pos
        out = pos.copy()
        w = self.risk_blocks_corr_window
        for bname, members in (self.risk_blocks_cfg or {}).items():
            mem = [a for a in (members or []) if a in traded
                   and a not in self.signal_only_assets]
            m = len(mem)
            if m < 2:
                continue
            pair_corrs = []
            for i in range(m):
                for j in range(i + 1, m):
                    pair_corrs.append(
                        self.dir_returns[mem[i]].rolling(w, min_periods=60)
                        .corr(self.dir_returns[mem[j]]))
            rho = (pd.concat(pair_corrs, axis=1).mean(axis=1)
                   .clip(0.0, 1.0).shift(1).reindex(pos.index))
            scale = (1.0 / np.sqrt(m * (1.0 + (m - 1) * rho))).fillna(1.0 / m)
            out[mem] = out[mem].mul(scale, axis=0)
        return out

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

    def _class_sleeves(self, assets: List[str], asset_class: str) -> Dict[str, pd.DataFrame]:
        """Active, neutralized sleeve signals for one asset class.

        Returns {sleeve_name: signal DataFrame} containing only sleeves with a
        non-zero weight and usable data, each already xs-neutralized by its own
        strength. Single source of truth shared by _combine_class (positions)
        and sleeve_snapshot (dashboard).
        """
        w = self.sleeve_weights

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

        # Rates-only directional sleeves (yield-driven; silently absent when
        # the yield maps have no data).
        if asset_class == 'rates':
            if w.get('curve', 0.0) != 0.0:
                cs = self.curve_signal(assets).reindex(columns=assets)
                if cs.notna().any().any():
                    sleeves['curve'] = cs
            if w.get('policy', 0.0) != 0.0:
                ps = self.policy_signal(assets).reindex(columns=assets)
                if ps.notna().any().any():
                    sleeves['policy'] = ps

        xs = self.xs_neutralize
        if asset_class == 'fx' and self.xs_neutralize_fx is not None:
            xs = self.xs_neutralize_fx
        out = {}
        for name, sig in sleeves.items():
            if w.get(name, 0.0) == 0.0 or sig is None or sig.empty:
                continue
            out[name] = self._xs_demean(sig.fillna(0.0), xs.get(name, 0.0))
        return out

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

        for name, sig in self._class_sleeves(assets, asset_class).items():
            wt = w.get(name, 0.0)
            combined = combined.add(wt * sig, fill_value=0.0)
            contrib += abs(wt)

        if contrib > 0:
            combined = combined / contrib
        if self.signal_smooth_span > 1:
            combined = combined.ewm(span=self.signal_smooth_span, min_periods=1).mean()
        return combined

    def sleeve_snapshot(self, asset_class: str = 'rates') -> Dict[str, Any]:
        """Latest-date view of the sleeve book for dashboards.

        Returns:
          {'date': Timestamp,
           'sleeves': {name: {'weight': float, 'xs_neutralize': float,
                              'signals': {asset: latest neutralized value}}},
           'target': {asset: final target position (vol-sized, vol-targeted,
                      position_smooth applied per config)}}
        """
        assets = self.rates_assets if asset_class == 'rates' else self.fx_assets
        sleeves_out = {}
        for name, sig in self._class_sleeves(assets, asset_class).items():
            sleeves_out[name] = {
                'weight': float(self.sleeve_weights.get(name, 0.0)),
                'xs_neutralize': float(self.xs_neutralize.get(name, 0.0)),
                'signals': {a: round(float(sig[a].iloc[-1]), 3) for a in sig.columns},
            }
        if asset_class == 'rates' and self.reversion_enabled:
            rsig = self._reversion_signal().iloc[-1]
            sleeves_out['reversion'] = {
                'weight': self.reversion_weight,
                'xs_neutralize': 1.0,
                'signals': {a: round(float(rsig.get(a, 0.0)), 3) for a in assets},
            }

        pos = self.finalize_positions(self.compute_target_positions())
        last = pos.iloc[-1]
        target = {a: round(float(last.get(a, 0.0)), 3) for a in assets}
        prev_row = pos.iloc[-2] if len(pos) >= 2 else last
        prev = {a: round(float(prev_row.get(a, 0.0)), 3) for a in assets}
        cols = [a for a in assets if a in pos.columns]

        # per-asset 1-day PnL contribution (fraction of book capital, ex-costs)
        pnl_last = (pos[cols].shift(1) * self.dir_returns[cols]).iloc[-1]
        pnl_1d = {a: float(pnl_last.get(a, 0.0)) for a in cols}

        return {'date': pos.index[-1], 'sleeves': sleeves_out, 'target': target,
                'prev': prev, 'pnl_1d': pnl_1d,
                # 시그널에는 기여하지만 주문은 내지 않는 자산 — 대시보드가
                # '중립(포지션 0)'과 '매매 대상 아님'을 구분해 표시하도록.
                'signal_only': sorted(a for a in assets if a in self.signal_only_assets),
                # recent per-asset position history (sparklines) + full net /
                # gross book series (delta- & gross-budget calibration)
                'history': pos[cols].tail(120),
                'net_hist': pos[cols].sum(axis=1),
                'gross_hist': pos[cols].abs().sum(axis=1)}

    # ──────────────────────────────────────────────────────────────────────
    def _mute_signal_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero the signal-only assets' columns (kept, so downstream code that
        iterates rates_assets still finds them — position/turnover/PnL = 0)."""
        cols = [c for c in df.columns if c in self.signal_only_assets]
        if not cols:
            return df
        df = df.copy()
        df[cols] = 0.0
        return df

    def _realized_vol(self) -> pd.DataFrame:
        """EWMA annualized realized vol per asset (directional returns)."""
        vol = self.dir_returns.ewm(halflife=self.vol_halflife, min_periods=20).std()
        return vol * np.sqrt(TRADING_DAYS)

    def compute_target_positions(self) -> pd.DataFrame:
        """Full pipeline → (dates × assets) directional target positions.

        커브 합성 자산(curve_trades)은 여기서 아웃라이트와 동급 '자산'으로
        사이징·볼타겟에 참여하고, 레그 환산은 finalize_positions 끝에서 일어난다.
        """
        traded = self.rates_assets + self.fx_assets + self.curve_assets
        conviction = pd.DataFrame(0.0, index=self.prices.index, columns=traded)

        if self.rates_assets:
            conviction[self.rates_assets] = self._combine_class(self.rates_assets, 'rates')
        if self.fx_assets:
            conviction[self.fx_assets] = self._combine_class(self.fx_assets, 'fx')
        if self.curve_assets:
            conviction[self.curve_assets] = self._combine_curve()

        conviction = conviction.fillna(0.0)
        # 시그널(위 _combine_class)은 전체 유니버스로 산출한 뒤, 매매 제외 자산만
        # 여기서 0으로 — 이후 인버스볼 사이징·포트폴리오 볼타겟이 남은 자산 기준
        # 으로 재계산되어 빠진 리스크 예산이 자동 재배분된다.
        conviction = self._mute_signal_only(conviction)

        # ── Inverse-vol sizing: equal risk per asset ─────────────────────
        vol = self._realized_vol()[traded]
        inv_vol = (self.target_asset_vol / vol).clip(upper=self.max_asset_pos * 5)
        pos = (conviction * inv_vol).clip(-self.max_asset_pos, self.max_asset_pos)
        pos = pos.fillna(0.0)

        # ── Block risk budgeting (no-op when disabled) ───────────────────
        pos = self._apply_risk_blocks(pos, traded)

        # ── Portfolio vol targeting (ex-ante, trailing) ──────────────────
        pos = self._apply_portfolio_vol_target(pos, traded)

        return pos

    def finalize_positions(self, pos: pd.DataFrame,
                           smooth_override: Optional[float] = None,
                           explode_curves: bool = True) -> pd.DataFrame:
        """Apply the production overlays to raw target positions, in order:
        ① position_smooth (EWMA, turnover damping) ② rates book stop
        ③ reversion sub-book blend ④ curve-leg 환산 ⑤ rates_exposure_scale.
        Single path shared by the backtest, the live feed and the dashboard
        snapshot so they can never diverge.

        explode_curves=False 는 진단용 — 커브 합성 컬럼을 그대로 남겨 커브 북
        단독 성과를 볼 때만 쓴다 (운용 경로는 항상 True = 실계약 레그).
        """
        smooth = self.position_smooth if smooth_override is None else float(smooth_override)
        if smooth > 0:
            pos = pos.ewm(alpha=1.0 - smooth, min_periods=1).mean()
        pos = self._apply_book_stop(pos)
        # ③ blend in the always-on reversion sub-book (rates only, un-stopped)
        if self.reversion_enabled and self.rates_assets:
            mr = self.reversion_positions()
            R = [a for a in self.rates_assets if a in pos.columns]
            w = self.reversion_weight
            pos = pos.copy()
            pos[R] = (1.0 - w) * pos[R] + w * mr[R]
        # ④ 커브 합성 → 실계약 레그 (스무딩·스톱이 합성 공간에서 끝난 뒤,
        #    노출 스케일 이전 — 레그도 금리 델타이므로 ⑤의 적용 대상이다).
        if explode_curves and self.curve_assets:
            pos = self._explode_curve_positions(pos)
        # ⑤ 금리 북 최종 노출 스케일 — 운용 북 대비 델타 캘리브레이션.
        #    스톱/리버전까지 끝난 최종 산출에만 곱하므로 시그널·스톱 동작은
        #    스케일 이전과 동일하고, 총/net 델타·PnL·계약수만 비례 축소된다.
        if self.rates_exposure_scale != 1.0 and self.rates_assets:
            R = [a for a in self.rates_assets if a in pos.columns]
            if not explode_curves:
                R = R + [a for a in self.curve_assets if a in pos.columns]
            if R:
                pos = pos.copy()
                pos[R] = pos[R] * self.rates_exposure_scale
        return pos

    def _reversion_signal(self) -> pd.DataFrame:
        """xs-demeaned fade of each rates market's `reversion_lookback`-day
        directional move (relative to the global-duration average)."""
        R = self.rates_assets
        fade = (-_zscore(self.dir_px[R].pct_change(self.reversion_lookback),
                         self.reversion_z_window)).clip(-self.signal_clip, self.signal_clip)
        return fade.sub(fade.mean(axis=1), axis=0)

    def reversion_positions(self) -> pd.DataFrame:
        """Fast market-neutral reversal book on rates (own vol targeting,
        light smoothing — heavier smoothing lags the 10d signal and loses
        more alpha than it saves in costs)."""
        R = self.rates_assets
        fade = self._reversion_signal()
        inv_vol = (self.target_asset_vol / self._realized_vol()[R]).clip(
            upper=self.max_asset_pos * 5)
        pos = (fade * inv_vol).clip(-self.max_asset_pos, self.max_asset_pos).fillna(0.0)
        # 페이드는 전체 금리 횡단면으로 계산(위)하고 매매만 제외 → 나머지 자산의
        # 상대가치 신호는 그대로 유지된다.
        pos = self._mute_signal_only(pos)
        pos = self._apply_portfolio_vol_target(pos, R)
        if self.reversion_smooth > 0:
            pos = pos.ewm(alpha=1.0 - self.reversion_smooth, min_periods=1).mean()
        return pos

    def _apply_book_stop(self, pos: pd.DataFrame) -> pd.DataFrame:
        """Shadow-equity trailing stop on the rates book (causal).

        Shadow PnL = what the un-stopped rates book would earn (pos.shift(1) ×
        dir_returns, gross). Its drawdown drives a scale applied with a 1-day
        lag: ≤dd_half → 1.0, ≤dd_flat → 0.5, beyond → 0.0. Using the SHADOW
        book (not the stopped one) means the stop re-engages on recovery
        instead of staying flat forever.

        Drawdown = 고점 대비 비율(%), 고점은 dd_window 일 롤링 최대 (0 = 전기간).
        롤링 고점이 재진입 경로다 — 전기간 고점을 쓰면 한 번 깊게 빠진 북이
        영원히 잠긴다 (2026-07-22 감사 참조).
        """
        if not self.book_stop_enabled or not self.rates_assets:
            return pos
        # 커브 합성도 금리 북의 일부 — 섀도우 PnL·스톱 스케일에 포함한다
        # (이 시점은 레그 환산 전이라 합성 컬럼이 살아 있다).
        R = [a for a in self.rates_assets + self.curve_assets if a in pos.columns]
        if not R:
            return pos
        shadow = (pos[R].shift(1).fillna(0.0) * self.dir_returns[R].fillna(0.0)).sum(axis=1)
        eq = (1.0 + shadow).cumprod()
        peak = (eq.cummax() if self.book_stop_dd_window <= 0
                else eq.rolling(self.book_stop_dd_window, min_periods=1).max())
        dd = (1.0 - eq / peak) * 100.0
        scale = pd.Series(
            np.select([dd <= self.book_stop_dd_half, dd <= self.book_stop_dd_flat],
                      [1.0, 0.5], 0.0),
            index=pos.index,
        ).shift(1).fillna(1.0)
        out = pos.copy()
        out[R] = out[R].mul(scale, axis=0)
        return out

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
