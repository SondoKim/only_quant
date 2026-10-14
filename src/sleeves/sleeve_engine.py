"""
Sleeve Engine — 금리 팩터 북의 연속·리스크 타게팅 포지션 (운용 엔진).

파이프라인 (전부 벡터화·인과 — t일 포지션은 t일 종가까지의 데이터만 쓴다.
scripts/invariants.py 의 절단 테스트가 이를 매일 실증한다):

  가격(롤 보정) ──► 방향성 수익률 (long = 듀레이션 롱 / 외화 롱)
        ──► 슬리브 시그널 (trend · value · carry · policy), 각각 연속 z
        ──► 슬리브별 횡단면 중립화 (xs_neutralize — 상대가치 슬리브만)
        ──► 슬리브 가중 결합 → 자산별 컨빅션 (+ EWMA 스무딩)
        ──► 시그널 전용 자산 뮤트 → 인버스볼 사이징 (자산당 동일 위험)
        ──► 북별 포트폴리오 볼 타게팅 (과거 수익률만 사용)
        ──► finalize: 포지션 스무딩 → 북스톱 → 노출 스케일 → DV01 한도

출력 포지션 = (날짜 × 자산) 방향성 공간의 '배정자본 대비 선물 명목 배수'.
손익 = positions.shift(1) × 방향성 수익률 (src/sleeves/accounting.py).

2026-10-13 정리: 기본 OFF·기각된 실험 기능(리버전 서브북, Hurst 레짐 게이트, 박스권/챱 북,
스위칭 토글, 인플레이션/정책경로 슬리브, 커브 슬리브·커브 트레이드, 블록 예산)을 운용
엔진에서 제거했다. 재현은 git tag `pre-engine-cleanup-2026-10`.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
DEFAULT_ASSETS_YAML = Path(__file__).parent.parent.parent / 'config' / 'assets.yaml'


def _zscore(df: pd.DataFrame, window: int, min_periods: Optional[int] = None) -> pd.DataFrame:
    """Rolling time-series z-score per column."""
    mp = min_periods or max(20, window // 3)
    mean = df.rolling(window, min_periods=mp).mean()
    std = df.rolling(window, min_periods=mp).std()
    return (df - mean) / std.replace(0.0, np.nan)


def rolling_dv01_beta(fut_returns: pd.Series, yield_pct: pd.Series,
                      lookback: int = 250, min_periods: int = 60, horizon: int = 5) -> pd.Series:
    """|β_t| = 명목 1 당 1bp 손익 (선물 horizon일 수익률을 같은 구간 금리 bp 변화에 회귀한 기울기의 크기).

    horizon 일 겹침 구간 변화로 회귀한다 (2026-10-14 사용자 결정 '5일 단위 회귀로 통일'): 일간 회귀는
    블룸버그 미국 금리 스냅샷이 CBOT 정산보다 늦어 β 가 낮게 나온다 (일간 상관 美 −0.90 vs 韓 −0.98 —
    TU 1.53 vs 5일 1.67, TY 5.40 vs 5.92). t일 값은 t일까지의 데이터만 쓴다 (인과).
    DV01 만원/bp = 포지션 × |β| × 배정자본(억) × 1e4 — 엔진 한도·대시보드·라이브 기록·PM 계약당 DV01 공용.
    """
    h = max(int(horizon), 1)
    r = fut_returns.rolling(h).sum() if h > 1 else fut_returns
    dy = yield_pct.diff(h) * 100.0
    b = (r.rolling(lookback, min_periods=min_periods).cov(dy)
         / dy.rolling(lookback, min_periods=min_periods).var())
    return b.abs()


class SleeveEngine:
    """Build continuous, risk-targeted sleeve positions from a price panel."""

    def __init__(
        self,
        prices: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
        assets_config_path: Optional[str] = None,
        yields: Optional[pd.DataFrame] = None,
        macro: Optional[pd.DataFrame] = None,
    ):
        self.prices = prices.sort_index()
        self.cfg = config or {}

        # 시그널 전용 패널 (매매 안 함): yields = carry·policy 슬리브, macro = 정책금리
        # (hedged_xs 캐리의 단기금리). 없으면 해당 슬리브가 조용히 비활성 → invariants 의
        # degen 점검이 '신호 없음'으로 잡는다.
        self.yields = None
        if yields is not None and not yields.empty:
            self.yields = yields.reindex(self.prices.index).ffill()
        self.macro = None
        if macro is not None and not macro.empty:
            self.macro = macro.reindex(self.prices.index).ffill()

        # ── Universes ────────────────────────────────────────────────────
        # exclude_assets: 유니버스에서 통째로 제외 (횡단면·볼타겟 어디에도 안 들어감).
        # signal_only_assets: 시그널·횡단면 기준선에는 참여, 최종 포지션만 0 (英日豪 —
        # 집행 시간 제약). 빠진 리스크는 볼타겟이 남은 자산으로 재배분한다.
        cols = list(self.prices.columns)
        excluded = set(self.cfg.get('exclude_assets', []) or [])
        self.rates_assets = [c for c in cols
                             if 'Comdty' in c and 'NQ' not in c and c not in excluded]
        self.fx_assets = [c for c in cols if 'Curncy' in c and c not in excluded]
        self.signal_only_assets = set(self.cfg.get('signal_only_assets', []) or [])

        # ── Sign map: long = 듀레이션 롱 / 외화 롱. KRW Curncy 는 USDKRW 현물(↑=원화 약세).
        self.sign = pd.Series(1.0, index=cols)
        if 'KRW Curncy' in self.sign.index:
            self.sign['KRW Curncy'] = -1.0
        for c, s in (self.cfg.get('sign_overrides', {}) or {}).items():
            if c in self.sign.index:
                self.sign[c] = float(s)

        acfg = self._load_assets_yaml(assets_config_path)
        sy = acfg.get('signal_yields', {}) or {}
        sm = acfg.get('signal_macro', {}) or {}
        # tradeable_yield_map: 선물 → 해당 테너 국채 금리 / fx_short_yield: FX → 2Y (+'US' 앵커)
        # policy_rate_map: 선물 → 자국 2Y/3Y (정책 프록시) / carry_short_rate_map: 선물 → 정책금리
        self.tradeable_yield_map = sy.get('tradeable_yield_map', {}) or {}
        self.fx_short_yield = sy.get('fx_short_yield', {}) or {}
        self.policy_rate_map = sy.get('policy_rate_map', {}) or {}
        self.carry_short_rate_map = sm.get('carry_short_rate_map', {}) or {}
        # 단기금리 보정(%p): FDTR = 연준 목표범위 '상단' → −0.125 로 중간값 (2026-10-14 사용자 결정 —
        # 다른 나라는 단일 정책금리라 미국만 캐리가 12.5bp 낮게 잡히던 정의 불일치)
        self.carry_short_rate_offset = {k: float(v) for k, v in
                                        (sm.get('carry_short_rate_offset', {}) or {}).items()}
        # 금리호가 선물(ASX: 100 − 금리) → 테너(년). 가격 수익률 변동성이 실제 채권 위험이 아니므로
        # 캐리 위험 분모는 par 수정듀레이션 × 금리 변동성으로 대체한다 (2026-10-13 감사).
        self.yield_quoted_tenor = {k: float(v) for k, v in
                                   (sy.get('yield_quoted_futures', {}) or {}).items()}
        self.carry_pairs = {p['fx']: {'us_rate': p.get('us_rate'),
                                      'foreign_rate': p.get('foreign_rate')}
                            for p in (acfg.get('carry_pairs', []) or [])
                            if p.get('fx') in self.prices.columns}

        # ── Parameters ───────────────────────────────────────────────────
        self.trend_horizons = self.cfg.get('trend_horizons', [126, 252])
        self.trend_z_window = self.cfg.get('trend_z_window', 252)
        self.value_window = self.cfg.get('value_window', 504)
        self.carry_window = self.cfg.get('carry_window', 252)
        self.signal_clip = self.cfg.get('signal_clip', 2.0)
        self.policy_period = int(self.cfg.get('policy_period', 126))
        # FX 캐리 (모니터링 전용): 'yield' = 2Y 금리차, 'price' = 금리선물 가격차 프록시.
        self.fx_carry_source = self.cfg.get('fx_carry_source', 'yield')
        # 금리 캐리: 'hedged_xs' = (테너금리 − 자국 정책금리) ÷ 채권 위험, 국가간 레벨 비교.
        #   (CIP: 환헤지가 펀딩을 US 단기로 바꾸므로 US 항은 상쇄.) 'legacy' 는 KE 캐리가
        #   US 2Y 와 비트 동일한 퇴화 산식 — 비교용으로만 남긴다.
        self.carry_method = str(self.cfg.get('carry_method', 'hedged_xs')).lower()
        self.sleeve_weights = self.cfg.get('sleeve_weights',
                                           {'trend': 1.0, 'value': 1.0, 'carry': 1.0, 'policy': 1.0})
        # 슬리브별 횡단면 중립화 강도 (1 = 시장중립 상대가치, 0 = 방향성 유지).
        xn = self.cfg.get('xs_neutralize', {'trend': 0.0, 'value': 1.0, 'carry': 1.0, 'policy': 0.0})
        if isinstance(xn, (int, float)):
            xn = {k: float(xn) for k in ('trend', 'value', 'carry', 'policy')}
        self.xs_neutralize = {k: float(v) for k, v in xn.items()}
        xnf = self.cfg.get('xs_neutralize_fx')
        self.xs_neutralize_fx = ({k: float(v) for k, v in xnf.items()}
                                 if isinstance(xnf, dict) else None)

        self.signal_smooth_span = int(self.cfg.get('signal_smooth_span', 5))   # 컨빅션 EWMA
        self.position_smooth = float(self.cfg.get('position_smooth', 0.0) or 0.0)   # 최종 포지션 EWMA

        # ── Book stop: 스톱 '전' 북(섀도우)의 손익으로 낙폭을 재서 금리 북을 감축/플랫.
        # 낙폭 = 고점 대비 비율(%), 고점 = dd_window 일 롤링 최대(0 = 전기간 → 래치 위험,
        # 2026-07-22 감사). 섀도우는 노출 스케일·DV01 한도 적용 '전' 포지션 기준이다.
        bs = self.cfg.get('book_stop', {}) or {}
        self.book_stop_enabled = bool(bs.get('enabled', False))
        self.book_stop_dd_half = float(bs.get('dd_half', 4.0))
        self.book_stop_dd_flat = float(bs.get('dd_flat', 8.0))
        self.book_stop_dd_window = int(bs.get('dd_window', 0) or 0)
        self._book_stop_state: Optional[Dict[str, pd.Series]] = None

        self.vol_halflife = self.cfg.get('vol_halflife', 33)
        self.target_asset_vol = self.cfg.get('target_asset_vol', 0.10)
        self.target_port_vol = self.cfg.get('target_port_vol', 0.10)
        self.port_vol_window = self.cfg.get('port_vol_window', 63)
        # 'separate' = 금리 북과 FX 북을 각자 타겟 (FX 는 모니터링 전용이라 서로 독립이어야 한다).
        self.vol_target_mode = str(self.cfg.get('vol_target_mode', 'separate')).lower()
        self.target_port_vol_rates = float(self.cfg.get('target_port_vol_rates', self.target_port_vol))
        self.target_port_vol_fx = float(self.cfg.get('target_port_vol_fx', self.target_port_vol))
        self.max_asset_pos = self.cfg.get('max_asset_pos', 3.0)
        self.max_gross_leverage = self.cfg.get('max_gross_leverage', 10.0)
        self.rates_exposure_scale = float(self.cfg.get('rates_exposure_scale', 1.0))

        # DV01 한도 (만원/bp, 배정자본 기준 — PM 배분 배수 적용 전). 순/그로스 DV01 이 넘으면
        # 그날 금리 포지션 전체를 비례 축소한다. 0/None = 비활성.
        dl = self.cfg.get('dv01_limit', {}) or {}
        self.dv01_capital = float(self.cfg.get('capital_억', dl.get('capital_억', 0.0)) or 0.0)
        self.dv01_net_limit = float(dl.get('net', 0.0) or 0.0)
        self.dv01_gross_limit = float(dl.get('gross', 0.0) or 0.0)
        self.dv01_lookback = int(dl.get('beta_lookback', 250))
        self.dv01_horizon = int(dl.get('beta_horizon', 5))
        self._dv01_state: Optional[Dict[str, Any]] = None

        # ── Derived series ───────────────────────────────────────────────
        self.dir_returns = self.prices.pct_change() * self.sign
        self.dir_px = (1.0 + self.dir_returns.fillna(0.0)).cumprod()

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_assets_yaml(path: Optional[str]) -> Dict[str, Any]:
        try:
            with open(path or DEFAULT_ASSETS_YAML, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _m(self, ticker: Optional[str]) -> Optional[pd.Series]:
        """Fetch a macro series by ticker, or None if unavailable."""
        if self.macro is None or ticker is None or ticker not in self.macro.columns:
            return None
        s = self.macro[ticker]
        return s if s.notna().any() else None

    def _y(self, ticker: Optional[str]) -> Optional[pd.Series]:
        """Fetch a yield series by ticker, or None if unavailable."""
        if self.yields is None or ticker is None or ticker not in self.yields.columns:
            return None
        s = self.yields[ticker]
        return s if s.notna().any() else None

    def _has_fx_yields(self) -> bool:
        us = self.fx_short_yield.get('US')
        return self._y(us) is not None and any(
            self._y(self.fx_short_yield.get(fx)) is not None for fx in self.fx_assets)

    def _has_rates_yields(self) -> bool:
        return any(self._y(self.tradeable_yield_map.get(a)) is not None
                   for a in self.rates_assets)

    # ──────────────────────────────────────────────────────────────────────
    # Sleeve signals (continuous, ~z-score scaled, directional space)
    # ──────────────────────────────────────────────────────────────────────
    def trend_signal(self, assets: List[str]) -> pd.DataFrame:
        """다중 호라이즌 시계열 모멘텀: h일 수익률의 롤링 z (trend_z_window) 평균."""
        px = self.dir_px[assets]
        parts = [_zscore(px.pct_change(h), self.trend_z_window) for h in self.trend_horizons]
        sig = pd.concat(parts).groupby(level=0).mean()
        return sig.clip(-self.signal_clip, self.signal_clip)

    def value_signal(self, assets: List[str]) -> pd.DataFrame:
        """장기 평균회귀: 로그 가격의 2년 롤링 z 의 음수 (비싸면 숏)."""
        px = self.dir_px[assets]
        z = _zscore(np.log(px.clip(lower=1e-9)), self.value_window)
        return (-z).clip(-self.signal_clip, self.signal_clip)

    def policy_signal(self, assets: List[str]) -> pd.DataFrame:
        """중앙은행 사이클 모멘텀: 자국 2Y/3Y 금리의 policy_period 일 하락폭 z (하락 = 롱)."""
        cols = {a: self._y(self.policy_rate_map.get(a)) for a in assets}
        cols = {a: y for a, y in cols.items() if y is not None}
        if not cols:
            return pd.DataFrame(index=self.prices.index, columns=assets)
        ydf = pd.DataFrame(cols)
        parts = [_zscore(-(ydf - ydf.shift(self.policy_period)), self.trend_z_window)]
        z = pd.concat(parts).groupby(level=0).mean()
        return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)

    def carry_signal(self, assets: List[str], asset_class: str) -> pd.DataFrame:
        """캐리 슬리브. FX(모니터링): 외국 2Y − US 2Y 의 z. 금리: carry_method 참조."""
        empty = pd.DataFrame(index=self.prices.index, columns=assets)
        if asset_class == 'fx':
            if self.fx_carry_source == 'yield' and self._has_fx_yields():
                us = self._y(self.fx_short_yield.get('US'))
                diffs = {fx: self._y(self.fx_short_yield.get(fx)) - us for fx in assets
                         if self._y(self.fx_short_yield.get(fx)) is not None}
                if diffs:
                    z = _zscore(pd.DataFrame(diffs), self.carry_window)
                    return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)
            diffs = {}
            for fx, rp in self.carry_pairs.items():
                ur, fr = rp.get('us_rate'), rp.get('foreign_rate')
                if fx in assets and ur in self.prices.columns and fr in self.prices.columns:
                    diffs[fx] = self.prices[ur] - self.prices[fr]
            if not diffs:
                return empty
            z = _zscore(pd.DataFrame(diffs), self.carry_window)
            return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)

        if not self._has_rates_yields():
            return empty
        if self.carry_method == 'hedged_xs':
            return self._carry_hedged_xs(assets)
        # legacy: hedged = local − (local 2Y/3Y − US 2Y), 자산별 시계열 z (비교용)
        us_fund = self._y(self.fx_short_yield.get('US'))
        ydf = {}
        for a in assets:
            ys = self._y(self.tradeable_yield_map.get(a))
            if ys is None:
                continue
            fund = self._y(self.policy_rate_map.get(a))
            ydf[a] = ys - (fund - us_fund) if (us_fund is not None and fund is not None) else ys
        if not ydf:
            return empty
        z = _zscore(pd.DataFrame(ydf), self.carry_window)
        return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)

    def carry_risk(self, assets: List[str]) -> pd.DataFrame:
        """캐리 위험 분모 (연 변동성, 채권 명목 기준).

        가격호가 선물 = 실현 수익률 변동성. 금리호가 선물(ASX, yield_quoted_futures) =
        par 수정듀레이션(테너, 금리) × 금리 변화 변동성 — 100−금리 호가의 가격 수익률은
        테너와 무관하게 듀레이션 ~1 처럼 움직여 위험을 최대 8배 과소평가한다.
        """
        vol = self._realized_vol()
        out = {}
        for a in assets:
            if a in self.yield_quoted_tenor:
                y = self._y(self.tradeable_yield_map.get(a))
                if y is None:
                    continue
                yd = y / 100.0
                T = self.yield_quoted_tenor[a]
                dmod = (1.0 - (1.0 + yd) ** (-T)) / yd / (1.0 + yd)
                sdy = yd.diff().ewm(halflife=self.vol_halflife, min_periods=20).std() * np.sqrt(TRADING_DAYS)
                out[a] = dmod * sdy
            elif a in vol.columns:
                out[a] = vol[a]
        return pd.DataFrame(out)

    def _carry_hedged_xs(self, assets: List[str]) -> pd.DataFrame:
        """국가간 환헤지 캐리: raw_i = (y_i − s_i)/100 ÷ 위험_i, 매일 횡단면 demean 후
        횡단면 RMS 의 롤링 평균(carry_window, 인과)으로 나눠 z 스케일. 3개국 미만이면 비활성."""
        empty = pd.DataFrame(index=self.prices.index, columns=assets)
        risk = self.carry_risk(assets)
        raw = {}
        for a in assets:
            ys = self._y(self.tradeable_yield_map.get(a))
            st = self.carry_short_rate_map.get(a)
            sr = self._m(st)
            if ys is None or sr is None or a not in risk.columns:
                continue
            sr = sr + self.carry_short_rate_offset.get(st, 0.0)
            raw[a] = (ys - sr) / 100.0 / risk[a].replace(0.0, np.nan)
        if len(raw) < 3:
            logger.warning("hedged_xs carry: 단기금리/일드가 있는 자산 %d개 < 3 — 캐리 비활성", len(raw))
            return empty
        df = pd.DataFrame(raw)
        dm = df.sub(df.mean(axis=1), axis=0)
        rms = np.sqrt((dm ** 2).mean(axis=1))
        scale = rms.rolling(self.carry_window, min_periods=60).mean()
        z = dm.div(scale.replace(0.0, np.nan), axis=0)
        return z.reindex(columns=assets).clip(-self.signal_clip, self.signal_clip)

    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _xs_demean(sig: pd.DataFrame, strength: float) -> pd.DataFrame:
        """Cross-sectionally demean each row by `strength` (market-neutral RV)."""
        if sig.empty or strength <= 0:
            return sig
        return sig - strength * sig.mean(axis=1).values.reshape(-1, 1)

    def _class_sleeves(self, assets: List[str], asset_class: str) -> Dict[str, pd.DataFrame]:
        """가중치 ≠ 0 이고 데이터가 있는 슬리브 → 각자의 강도로 xs 중립화된 시그널.
        포지션(_combine_class)과 대시보드(sleeve_snapshot)가 공유하는 단일 출처."""
        w = self.sleeve_weights
        sleeves = {'trend': self.trend_signal(assets),
                   'value': self.value_signal(assets)}
        if asset_class == 'fx' or self._has_rates_yields():
            sleeves['carry'] = self.carry_signal(assets, asset_class).reindex(columns=assets)
        if asset_class == 'rates' and w.get('policy', 0.0) != 0.0:
            ps = self.policy_signal(assets).reindex(columns=assets)
            if ps.notna().any().any():
                sleeves['policy'] = ps
        xs = self.xs_neutralize
        if asset_class == 'fx' and self.xs_neutralize_fx is not None:
            xs = self.xs_neutralize_fx
        return {name: self._xs_demean(sig.fillna(0.0), xs.get(name, 0.0))
                for name, sig in sleeves.items()
                if w.get(name, 0.0) != 0.0 and sig is not None and not sig.empty}

    def _combine_class(self, assets: List[str], asset_class: str) -> pd.DataFrame:
        """슬리브 가중 평균 컨빅션 (+ signal_smooth_span EWMA)."""
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

    # ──────────────────────────────────────────────────────────────────────
    def _mute_signal_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """시그널 전용 자산 컬럼을 0 으로 (컬럼은 유지 → 하류 코드가 그대로 순회 가능)."""
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
        """컨빅션 → 인버스볼 사이징 → 북별 포트폴리오 볼 타게팅 (finalize 전 원 포지션)."""
        traded = self.rates_assets + self.fx_assets
        conviction = pd.DataFrame(0.0, index=self.prices.index, columns=traded)
        if self.rates_assets:
            conviction[self.rates_assets] = self._combine_class(self.rates_assets, 'rates')
        if self.fx_assets:
            conviction[self.fx_assets] = self._combine_class(self.fx_assets, 'fx')
        conviction = self._mute_signal_only(conviction.fillna(0.0))

        vol = self._realized_vol()[traded]
        inv_vol = (self.target_asset_vol / vol).clip(upper=self.max_asset_pos * 5)
        pos = (conviction * inv_vol).clip(-self.max_asset_pos, self.max_asset_pos).fillna(0.0)

        if self.vol_target_mode == 'separate':
            out = pos.copy()
            for cols, tgt in ((self.rates_assets, self.target_port_vol_rates),
                              (self.fx_assets, self.target_port_vol_fx)):
                cols = [a for a in cols if a in pos.columns]
                if cols:
                    out[cols] = self._apply_portfolio_vol_target(pos[cols], cols, target=tgt)
            return out
        return self._apply_portfolio_vol_target(pos, traded)

    def finalize_positions(self, pos: pd.DataFrame,
                           smooth_override: Optional[float] = None) -> pd.DataFrame:
        """운용 오버레이를 순서대로: ① position_smooth ② 금리 북스톱 ③ 금리 노출 스케일
        ④ DV01 한도. 백테스트·라이브 피드·대시보드가 모두 이 경로 하나를 쓴다."""
        smooth = self.position_smooth if smooth_override is None else float(smooth_override)
        if smooth > 0:
            pos = pos.ewm(alpha=1.0 - smooth, min_periods=1).mean()
        pos = self._apply_book_stop(pos)
        R = [a for a in self.rates_assets if a in pos.columns]
        if self.rates_exposure_scale != 1.0 and R:
            pos = pos.copy()
            pos[R] = pos[R] * self.rates_exposure_scale
        return self._apply_dv01_limit(pos)

    def _apply_book_stop(self, pos: pd.DataFrame) -> pd.DataFrame:
        """섀도우(스톱 전) 금리 북 손익의 롤링 고점 대비 낙폭 → 다음날 배율 1 / 0.5 / 0 (인과)."""
        if not self.book_stop_enabled or not self.rates_assets:
            return pos
        R = [a for a in self.rates_assets if a in pos.columns]
        if not R:
            return pos
        shadow = (pos[R].shift(1).fillna(0.0) * self.dir_returns[R].fillna(0.0)).sum(axis=1)
        eq = (1.0 + shadow).cumprod()
        peak = (eq.cummax() if self.book_stop_dd_window <= 0
                else eq.rolling(self.book_stop_dd_window, min_periods=1).max())
        dd = (1.0 - eq / peak) * 100.0
        scale = pd.Series(
            np.select([dd <= self.book_stop_dd_half, dd <= self.book_stop_dd_flat], [1.0, 0.5], 0.0),
            index=pos.index,
        ).shift(1).fillna(1.0)
        self._book_stop_state = {'dd': dd, 'scale': scale}
        out = pos.copy()
        out[R] = out[R].mul(scale, axis=0)
        return out

    def dv01_betas(self, assets: Optional[List[str]] = None) -> pd.DataFrame:
        """자산별 롤링 |β_t| (명목 1 당 1bp 손익). 일드가 없는 자산은 빠진다."""
        out = {}
        for a in (assets or self.rates_assets):
            y = self._y(self.tradeable_yield_map.get(a))
            if y is None or a not in self.dir_returns.columns or a in self.yield_quoted_tenor:
                continue
            out[a] = rolling_dv01_beta(self.dir_returns[a], y, self.dv01_lookback,
                                       horizon=self.dv01_horizon)
        return pd.DataFrame(out, index=self.prices.index)

    def _apply_dv01_limit(self, pos: pd.DataFrame) -> pd.DataFrame:
        """DV01 한도: 순 |Σ DV01| > net 또는 Σ|DV01| > gross 이면 그날 금리 포지션을 비례 축소.
        DV01(만원/bp) = 포지션 × |β_t| × capital(억) × 1e4. β_t 는 t일까지만 사용 (인과)."""
        if self.dv01_capital <= 0 or (self.dv01_net_limit <= 0 and self.dv01_gross_limit <= 0):
            return pos
        R = [a for a in self.rates_assets if a in pos.columns and a not in self.signal_only_assets]
        k = self.dv01_betas(R).reindex(pos.index).ffill() * self.dv01_capital * 1e4
        R = [a for a in R if a in k.columns]
        if not R:
            return pos
        dv = pos[R] * k[R]
        net, gross = dv.sum(axis=1).abs(), dv.abs().sum(axis=1)
        f = pd.Series(1.0, index=pos.index)
        if self.dv01_net_limit > 0:
            f = np.minimum(f, (self.dv01_net_limit / net.replace(0.0, np.nan)).fillna(1.0))
        if self.dv01_gross_limit > 0:
            f = np.minimum(f, (self.dv01_gross_limit / gross.replace(0.0, np.nan)).fillna(1.0))
        f = f.clip(upper=1.0)
        self._dv01_state = {'factor': f, 'net_before': dv.sum(axis=1), 'gross_before': gross}
        out = pos.copy()
        RR = [a for a in self.rates_assets if a in pos.columns]
        out[RR] = out[RR].mul(f, axis=0)
        return out

    def book_stop_status(self) -> Optional[Dict[str, Any]]:
        """마지막 실행 기준 북스톱 상태 (대시보드/해석용, 없으면 None).

        scale = 오늘 포지션에 적용된 배율 — 전일 낙폭(dd_prev)으로 정해진다 (shift(1), 인과).
        dd = 오늘 종가 낙폭 → next_scale = 다음 거래일에 적용될 배율. 임계를 넘는 날 '오늘 낙폭'과
        '오늘 배율'의 기준일이 하루 어긋나 보이던 문구(2026-10-14 검토)를 두 값으로 나눠 해소."""
        st = self._book_stop_state
        if not self.book_stop_enabled or not st:
            return None
        dd = st['dd']

        def stage(x: float) -> float:
            return 1.0 if x <= self.book_stop_dd_half else (0.5 if x <= self.book_stop_dd_flat else 0.0)
        d_now = float(dd.iloc[-1])
        d_prev = float(dd.iloc[-2]) if len(dd) >= 2 else float('nan')
        return {'scale': float(st['scale'].iloc[-1]), 'dd': d_now, 'dd_prev': d_prev,
                'next_scale': stage(d_now),
                'dd_half': self.book_stop_dd_half, 'dd_flat': self.book_stop_dd_flat,
                'dd_window': self.book_stop_dd_window}

    def _apply_portfolio_vol_target(self, pos: pd.DataFrame, traded: List[str],
                                    target: Optional[float] = None) -> pd.DataFrame:
        """매일 북 전체를 스케일해 과거 port_vol_window 일 포트폴리오 변동성 ≈ target.
        오늘 가중치 × '과거' 수익률만 사용 (오늘 수익률 미포함). pos 컬럼 순서 = traded 순서."""
        tgt = self.target_port_vol if target is None else float(target)
        ret_vals = self.dir_returns[traded].fillna(0.0).values
        pos_vals = pos.values
        win = self.port_vol_window
        scales = np.ones(len(pos))
        scales[:win] = 0.0                                  # 워밍업: 플랫
        for i in range(win, len(pos)):
            w = pos_vals[i]
            if not np.any(w):
                continue
            pv = (ret_vals[i - win:i] @ w).std() * np.sqrt(TRADING_DAYS)
            if np.isfinite(pv) and pv > 1e-9:
                scales[i] = min(tgt / pv, self.max_gross_leverage / (np.abs(w).sum() + 1e-9))
            else:
                scales[i] = 0.0   # 변동성 추정 불가(데이터 이상) → 스케일 1 로 방치하지 않고 플랫
        scaled = pos.mul(scales, axis=0)
        gross = scaled.abs().sum(axis=1)
        over = gross > self.max_gross_leverage
        if over.any():
            scaled.loc[over] = scaled.loc[over].mul(self.max_gross_leverage / gross[over], axis=0)
        return scaled.fillna(0.0)

    # ──────────────────────────────────────────────────────────────────────
    def sleeve_snapshot(self, asset_class: str = 'rates') -> Dict[str, Any]:
        """대시보드용 최신일 스냅샷: 슬리브 시그널·목표/전일 포지션·전일 손익·북스톱·이력."""
        assets = self.rates_assets if asset_class == 'rates' else self.fx_assets
        sleeves_out = {}
        for name, sig in self._class_sleeves(assets, asset_class).items():
            sleeves_out[name] = {
                'weight': float(self.sleeve_weights.get(name, 0.0)),
                'xs_neutralize': float(self.xs_neutralize.get(name, 0.0)),
                'signals': {a: round(float(sig[a].iloc[-1]), 3) for a in sig.columns},
            }
        pos = self.finalize_positions(self.compute_target_positions())
        last = pos.iloc[-1]
        target = {a: round(float(last.get(a, 0.0)), 3) for a in assets}
        prev_row = pos.iloc[-2] if len(pos) >= 2 else last
        prev = {a: round(float(prev_row.get(a, 0.0)), 3) for a in assets}
        cols = [a for a in assets if a in pos.columns]
        pnl_last = (pos[cols].shift(1) * self.dir_returns[cols]).iloc[-1]
        return {'date': pos.index[-1], 'sleeves': sleeves_out, 'target': target,
                'prev': prev, 'pnl_1d': {a: float(pnl_last.get(a, 0.0)) for a in cols},
                'book_stop': self.book_stop_status(),
                'dv01_factor': (float(self._dv01_state['factor'].iloc[-1])
                                if self._dv01_state else None),
                'sleeve_weights': dict(self.sleeve_weights),
                'signal_only': sorted(a for a in assets if a in self.signal_only_assets),
                'history': pos[cols].tail(120),
                'positions': pos[cols],
                'net_hist': pos[cols].sum(axis=1),
                'gross_hist': pos[cols].abs().sum(axis=1)}
