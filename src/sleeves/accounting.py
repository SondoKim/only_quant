"""
금리 북 손익 회계 — 단일 출처 (2026-10-13).

예전에는 같은 회계가 15개 스크립트에 복사돼 있었고, 사본마다 손익 기준(선물 수익률 vs
금리환산)이 달라 2026-10-02 감사의 '부풀린 SR' 이 생겼다. 백테스트·감사·연구 스크립트는
반드시 여기 함수만 쓴다.

규약
  · 포지션 pos[t] = t일 종가에 결정한 목표 (배정자본 대비 명목 배수, 방향성 공간)
  · 손익 pnl[t] = Σ pos[t − lag] × 방향성 수익률[t] − 비용(|Δpos[t]| × bps)
    lag 기본 1 (종가 결정 → 다음 구간 보유). 시차 감사는 lag_map 으로 자산별 지연.
  · basis 'futures' = 롤 보정 선물 수익률 (실계좌 손익; 운용 기준)
          'yield_implied' = 금리 bp 변화 × 롤링 β (구방식, 캐리·베이시스 누락 — 비교용)
"""
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# 자산별 거래비용 (명목 회전율 대비 bp)
DEFAULT_COSTS_BPS = {'rates': 0.5, 'fx': 1.0, 'index': 1.0, 'KRW Curncy': 3.0}


def cost_bps_for(asset: str, costs: Optional[Dict[str, float]] = None) -> float:
    costs = costs or DEFAULT_COSTS_BPS
    if asset in costs:
        return costs[asset]
    if 'Comdty' in asset:
        return costs.get('rates', 0.5)
    if 'Curncy' in asset:
        return costs.get('fx', 1.0)
    return costs.get('index', 1.0)


def engine_costs(engine) -> Dict[str, float]:
    return {**DEFAULT_COSTS_BPS, **((engine.cfg or {}).get('costs_bps', {}) or {})}


def asset_returns(engine, assets: Iterable[str], index: pd.Index,
                  basis: str = 'futures', beta_lookback: int = 250) -> pd.DataFrame:
    """손익용 자산 수익률. yield_implied 는 β 를 하루 시프트(인과), 결측은 선물 수익률로 폴백."""
    assets = list(assets)
    rets = engine.dir_returns[assets].reindex(index).fillna(0.0)
    if str(basis).lower() == 'futures':
        return rets
    out = rets.copy()
    for a in assets:
        yt = engine.tradeable_yield_map.get(a)
        if engine.yields is None or yt is None or yt not in engine.yields.columns:
            continue
        dy = engine.yields[yt].reindex(index).diff() * 100.0
        beta = (rets[a].rolling(beta_lookback, min_periods=60).cov(dy)
                / dy.rolling(beta_lookback, min_periods=60).var()).shift(1)
        imp = beta * dy
        out[a] = imp.where(imp.notna(), rets[a])
    return out


def asset_pnl(engine, pos: pd.DataFrame, assets: Optional[Iterable[str]] = None,
              basis: Optional[str] = None, lag_map: Optional[Dict[str, int]] = None,
              cost_mult: float = 1.0) -> pd.DataFrame:
    """자산별 일별 순손익 (배정자본 대비 비율). assets 기본 = 엔진 금리 유니버스."""
    assets = [a for a in (assets if assets is not None else engine.rates_assets) if a in pos.columns]
    basis = basis or (engine.cfg or {}).get('pnl_basis', 'futures')
    rets = asset_returns(engine, assets, pos.index, basis)
    held = pd.DataFrame({a: pos[a].shift((lag_map or {}).get(a, 1)) for a in assets},
                        index=pos.index).fillna(0.0)
    costs = engine_costs(engine)
    crate = pd.Series({a: cost_bps_for(a, costs) / 1e4 for a in assets})
    turnover = pos[assets].diff().abs().fillna(0.0)
    return held * rets - turnover.mul(crate, axis=1) * cost_mult


def book_pnl(engine, pos: pd.DataFrame, assets: Optional[Iterable[str]] = None,
             basis: Optional[str] = None, lag_map: Optional[Dict[str, int]] = None,
             cost_mult: float = 1.0) -> pd.Series:
    """북 전체 일별 순손익 (asset_pnl 의 합)."""
    return asset_pnl(engine, pos, assets, basis, lag_map, cost_mult).sum(axis=1)


# ── 성과 지표 ────────────────────────────────────────────────────────────
def window(s: pd.Series, start=None, end=None) -> pd.Series:
    if start is not None:
        s = s[s.index >= pd.to_datetime(start)]
    if end is not None:
        s = s[s.index <= pd.to_datetime(end)]
    return s


def sharpe(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean() / s.std() * np.sqrt(TRADING_DAYS)) if len(s) > 1 and s.std() > 0 else 0.0


def max_drawdown(s: pd.Series, compound: bool = True) -> float:
    """최대 낙폭. compound=True 는 (1+r) 복리 자산곡선, False 는 단순합(%p)."""
    s = s.fillna(0.0)
    eq = (1 + s).cumprod() if compound else s.cumsum()
    return float((eq / eq.cummax() - 1).min()) if compound else float((eq - eq.cummax()).min())


def perf_stats(daily: pd.Series) -> dict:
    daily = daily.dropna()
    if daily.empty or daily.std() == 0:
        return {'sharpe': 0.0, 'vol': 0.0, 'ret': 0.0, 'maxdd': 0.0}
    eq = (1 + daily).cumprod()
    return {'sharpe': sharpe(daily), 'vol': float(daily.std() * np.sqrt(TRADING_DAYS)),
            'ret': float(eq.iloc[-1] ** (TRADING_DAYS / len(daily)) - 1),
            'maxdd': max_drawdown(daily)}
