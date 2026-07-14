"""
Sleeve-engine backtest.

Runs the continuous, risk-targeted sleeve book (Trend / Value / Carry) and
reports performance vs the legacy strategy-factory. Because every signal is
rolling and NOTHING is fitted, a single full-sample pass with positions
shifted by one day is already out-of-sample at every point (no walk-forward
folder machinery needed).

Usage:
    python scripts/run_sleeve_backtest.py
    python scripts/run_sleeve_backtest.py --start-date 2022-01-01 --target-vol 0.10
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.sleeves.sleeve_engine import SleeveEngine, TRADING_DAYS

# Default per-asset transaction cost (basis points of notional turnover)
DEFAULT_COSTS_BPS = {'rates': 0.5, 'fx': 1.0, 'index': 1.0, 'KRW Curncy': 3.0}


def load_sleeve_config() -> dict:
    path = Path(__file__).parent.parent / 'config' / 'indicators.yaml'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return (yaml.safe_load(f) or {}).get('sleeves', {}) or {}
    return {}


def cost_bps_for(asset: str, costs: dict) -> float:
    if asset in costs:
        return costs[asset]
    if 'Comdty' in asset:
        return costs.get('rates', 0.5)
    if 'Curncy' in asset:
        return costs.get('fx', 1.0)
    return costs.get('index', 1.0)


def save_factor_signals(engine, out_path='sleeve_factor_signals.csv'):
    """Persist the rates factor z-score time-series (Trend/Value/Carry/Curve/
    Policy) exactly as the SleeveEngine computes them — the authoritative source
    for total_dashboard's '금리 팩터 모니터링' chart.

    Saved wide, one column per factor×asset ('{factor}::{ticker}'), matching the
    'pos::' / 'rpnl::' prefix convention in sleeve_backtest_log.csv. Full history
    is written (the dashboard slices to its own display start on read). This lets
    the dashboard READ these instead of re-deriving them, so changing the engine
    here can never silently desync the chart.

    NOTE: 모니터링 유니버스 = 가격 패널의 전체 금리자산 (exclude_assets 로 거래에서
    빠진 자산 포함 — 유로존 등 비거래 시장도 팩터 관찰 가치가 있어 차트에는 유지).
    거래 포지션(pos::)은 sleeve_backtest_log.csv 쪽이 축소 유니버스 기준.
    """
    ra = [c for c in engine.prices.columns if 'Comdty' in c and 'NQ' not in c]
    if not ra:
        print("⚠ No rates assets — factor signals not saved.")
        return None
    factors = {
        'trend':  engine.trend_signal(ra),
        'value':  engine.value_signal(ra, 'rates'),
        'carry':  engine.carry_signal(ra, 'rates'),
        'curve':  engine.curve_signal(ra),
        'policy': engine.policy_signal(ra),
    }
    blocks = []
    for fac, df in factors.items():
        if df is None or df.empty:
            continue
        d = df.reindex(columns=ra).dropna(how='all')
        if d.empty:
            continue
        blocks.append(d.rename(columns={a: f'{fac}::{a}' for a in d.columns}))
    if not blocks:
        print("⚠ Factor signals all empty — nothing saved.")
        return None
    out = pd.concat(blocks, axis=1).sort_index()
    out.index.name = 'date'
    out.to_csv(out_path)
    facs = sorted({c.split('::', 1)[0] for c in out.columns})
    print(f"✅ Saved {out_path}  ({out.shape[0]} rows × {out.shape[1]} cols; "
          f"factors: {', '.join(facs)})")
    return out


def perf_stats(daily: pd.Series) -> dict:
    daily = daily.dropna()
    if daily.empty or daily.std() == 0:
        return {'sharpe': 0.0, 'vol': 0.0, 'ret': 0.0, 'maxdd': 0.0}
    sharpe = daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    eq = (1 + daily).cumprod()
    ann_ret = eq.iloc[-1] ** (TRADING_DAYS / len(daily)) - 1
    maxdd = (eq / eq.cummax() - 1).min()
    return {'sharpe': sharpe, 'vol': vol, 'ret': ann_ret, 'maxdd': maxdd}


def run(start_date=None, end_date=None, target_vol=None, smooth=0.0, plot=True,
        data_start='2010-01-01', cfg_override=None, exclude_assets=None,
        save_outputs=True):
    print("📊 Loading price data...")
    loader = DataLoader()
    prices = DataPreprocessor(loader.load_data(start_date=data_start,
                                               use_cache=True)).clean().get_data()
    if exclude_assets:
        # 유니버스 축소 실험/운용: 엔진 유니버스는 prices 컬럼에서 결정되므로
        # 여기서 제외하면 시그널·xs-demean·볼타겟팅 모두 축소 유니버스로 재계산됨.
        drop = [a for a in exclude_assets if a in prices.columns]
        prices = prices.drop(columns=drop)
        print(f"   ⛔ Excluded {len(drop)} assets: {', '.join(drop)}")
    if start_date:
        prices = prices[prices.index >= pd.to_datetime(start_date)]
    if end_date:
        prices = prices[prices.index <= pd.to_datetime(end_date)]

    # Signal-only yields (Carry/Value/Curve/Policy). Fixed 2010+ cache so the
    # window never triggers per-run Bloomberg fetches.
    yields = loader.load_signal_yields(start_date="2010-01-01", use_cache=True)

    cfg = load_sleeve_config()
    if cfg_override:
        for k, v in cfg_override.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    if target_vol is not None:
        cfg['target_port_vol'] = target_vol
    costs = {**DEFAULT_COSTS_BPS, **(cfg.get('costs_bps', {}) or {})}

    engine = SleeveEngine(prices, config=cfg, yields=yields)
    src_fx = 'yields' if engine._has_fx_yields() else 'price-proxy'
    src_rt = 'yields' if engine._has_rates_yields() else 'OFF (no yields)'
    print(f"   Rates: {len(engine.rates_assets)} | FX: {len(engine.fx_assets)} | "
          f"target portfolio vol = {engine.target_port_vol:.0%}")
    print(f"   Carry source — FX: {src_fx} | Rates carry/value: {src_rt}")

    # Overlays (position_smooth + rates book stop) applied inside the engine —
    # same path as the live feed and dashboard. CLI --smooth overrides config.
    positions = engine.finalize_positions(
        engine.compute_target_positions(),
        smooth_override=smooth if smooth and smooth > 0 else None,
    )
    if engine.book_stop_enabled:
        print(f"   🛑 Rates book stop ON (shadow-DD half {engine.book_stop_dd_half:.0f}% "
              f"/ flat {engine.book_stop_dd_flat:.0f}%)")

    traded = list(positions.columns)
    dir_returns = engine.dir_returns[traded].reindex(positions.index).fillna(0.0)

    # 금리 자산 손익은 연속선물({ticker}1) 수익률 대신 '캐시금리 변동(bp) × 회귀 베타'로
    # 환산한 이론 수익률로 계산한다 → 연속선물 롤 점프(아티팩트) 제거.
    # 포지션 사이징은 그대로 실제 선물수익률 기반이고, 손익 귀속만 금리 기준으로 바꾼다.
    # beta = d(directional return)/d(yield bp), 최근 LB일 회귀(룩어헤드 방지로 1일 시프트).
    rates_cols = engine.rates_assets
    pnl_returns = dir_returns.copy()
    LB = 250
    yblk = engine.yields
    for a in rates_cols:
        if a not in pnl_returns.columns:
            continue
        yt = engine.tradeable_yield_map.get(a)
        if yblk is None or yt is None or yt not in yblk.columns:
            continue
        dy_bp = yblk[yt].reindex(dir_returns.index).diff() * 100.0
        fr = dir_returns[a]
        beta = (fr.rolling(LB, min_periods=60).cov(dy_bp)
                / dy_bp.rolling(LB, min_periods=60).var()).shift(1)
        implied = beta * dy_bp
        # 베타 워밍업 전이나 금리 결측 구간은 실제 수익률로 폴백.
        pnl_returns[a] = implied.where(implied.notna(), dir_returns[a])

    # PnL: yesterday's position earns today's directional return
    held = positions.shift(1).fillna(0.0)
    gross_pnl = held * pnl_returns

    # Transaction costs on daily turnover
    turnover = positions.diff().abs().fillna(0.0)
    cost_rate = pd.Series({a: cost_bps_for(a, costs) / 10000.0 for a in traded})
    cost_pnl = turnover.mul(cost_rate, axis=1)
    net_pnl = gross_pnl - cost_pnl

    rates_cols = engine.rates_assets
    fx_cols = engine.fx_assets

    port = net_pnl.sum(axis=1)
    rates = net_pnl[rates_cols].sum(axis=1) if rates_cols else pd.Series(0.0, index=port.index)
    fx = net_pnl[fx_cols].sum(axis=1) if fx_cols else pd.Series(0.0, index=port.index)

    # ── Report ───────────────────────────────────────────────────────────
    avg_gross = positions.abs().sum(axis=1).mean()
    ann_turnover = turnover.sum(axis=1).mean() * TRADING_DAYS
    print("\n" + "=" * 64)
    print(f"  SLEEVE BACKTEST  {positions.index[0].date()} → {positions.index[-1].date()}")
    print("=" * 64)
    for name, series in [('PORTFOLIO', port), ('  Rates', rates), ('  FX', fx)]:
        s = perf_stats(series)
        print(f"{name:<12} Sharpe {s['sharpe']:5.2f} | Vol {s['vol']:5.1%} | "
              f"AnnRet {s['ret']:6.1%} | MaxDD {s['maxdd']:6.1%}")
    print(f"\n  Avg gross leverage: {avg_gross:.2f}x | Annualized turnover: {ann_turnover:.1f}x")

    # ── Plot ───────────────────────────────────────────────────────────────
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[3, 1])
        for series, label in [(port, 'Portfolio'), (rates, 'Rates'), (fx, 'FX')]:
            s = perf_stats(series)
            (1 + series).cumprod().plot(ax=ax[0], label=f"{label} (SR {s['sharpe']:.2f})")
        ax[0].set_title(f"Sleeve Engine — Trend/Value/Carry "
                        f"(target vol {engine.target_port_vol:.0%})")
        ax[0].set_ylabel("Cumulative (1+r)"); ax[0].grid(alpha=0.3); ax[0].legend()
        positions.abs().sum(axis=1).plot(ax=ax[1], color='gray', lw=0.8)
        ax[1].set_ylabel("Gross lev"); ax[1].grid(alpha=0.3)
        sd = positions.index[0].strftime('%Y-%m-%d')
        out = f"sleeve_backtest_{sd}_tv{int(engine.target_port_vol*100)}.png"
        plt.tight_layout(); plt.savefig(out, dpi=110)
        print(f"\n📊 Saved plot → {out}")

    # Save daily series for further analysis
    out_df = pd.DataFrame({'portfolio': port, 'rates': rates, 'fx': fx})
    out_df['equity'] = (1 + port).cumprod()
    # 자산별 금리 일별 순손익 (대시보드 '금리 자산별 누적손익(억원)' 차트용).
    # 컬럼 합 = 'rates' 와 동일. prefix 'rpnl::' 로 기존 컬럼과 구분.
    for a in rates_cols:
        out_df[f'rpnl::{a}'] = net_pnl[a]
    # 자산별 일별 포지션 (대시보드 '총/net 델타 시계열' 차트용). prefix 'pos::'.
    for a in rates_cols:
        out_df[f'pos::{a}'] = positions[a]
    if save_outputs:
        out_df.to_csv('sleeve_backtest_log.csv')
        print("✅ Saved sleeve_backtest_log.csv")

        # 대시보드 '금리 팩터 모니터링' 소스 — 엔진 산출 팩터 z-score를 그대로 저장.
        save_factor_signals(engine)

    return out_df


def main():
    p = argparse.ArgumentParser(description="Sleeve engine backtest")
    p.add_argument('--start-date', default='2022-01-01')
    p.add_argument('--end-date', default=None)
    p.add_argument('--target-vol', type=float, default=None,
                   help='Target annualized portfolio vol (overrides config)')
    p.add_argument('--smooth', type=float, default=0.0,
                   help='EWMA position smoothing 0..1 (fraction of prior held); 0=off')
    p.add_argument('--data-start', default='2010-01-01',
                   help='Price-panel load start (loader truncates to common inception)')
    p.add_argument('--config-json', default=None,
                   help='JSON dict merged over the sleeves config, e.g. '
                        '\'{"sleeve_weights": {"curve": 1.0}}\'')
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()
    import json
    cfg_override = json.loads(args.config_json) if args.config_json else None
    run(start_date=args.start_date, end_date=args.end_date,
        target_vol=args.target_vol, smooth=args.smooth, plot=not args.no_plot,
        data_start=args.data_start, cfg_override=cfg_override)


if __name__ == "__main__":
    main()
