"""
금리 북 백테스트 (SleeveEngine) → sleeve_backtest_log.csv · sleeve_factor_signals.csv.

모든 시그널이 롤링·무적합이라 포지션을 하루 시프트한 단일 전체 표본 패스가 곧 매 시점
표본외다 (절단 테스트로 실증 — scripts/invariants.py). 손익 회계는 src/sleeves/accounting.py.
주의: 하이퍼파라미터·슬리브 구성은 같은 표본에서 여러 번 선택됐으므로 보고 SR 은 낙관적이다.

Usage:
    python scripts/run_sleeve_backtest.py --start-date 2016-01-01
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
from src.sleeves.sleeve_engine import SleeveEngine, TRADING_DAYS
# 손익 회계 단일 출처 — 아래 이름들은 하위 호환용 재수출
from src.sleeves.accounting import (asset_pnl, book_pnl, perf_stats, cost_bps_for,  # noqa: F401
                                    DEFAULT_COSTS_BPS, sharpe, max_drawdown, window)


def load_sleeve_config() -> dict:
    path = Path(__file__).parent.parent / 'config' / 'indicators.yaml'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return (yaml.safe_load(f) or {}).get('sleeves', {}) or {}
    return {}


def freshness_watch(engine) -> list:
    """신선도 감시 티커: 시그널 유니버스 금리선물 + 그 일드 + USDKRW(계약 환산)."""
    w = list(engine.rates_assets) + ['KRW Curncy']
    for a in engine.rates_assets:
        for m in (engine.tradeable_yield_map, engine.policy_rate_map):
            if m.get(a):
                w.append(m[a])
    return sorted(set(w))


def save_factor_signals(engine, out_path='sleeve_factor_signals.csv'):
    """Persist the rates factor z-score time-series (Trend/Value/Carry/Curve/
    Policy) exactly as the SleeveEngine computes them — the authoritative source
    for total_dashboard's '금리 팩터 모니터링' chart.

    Saved wide, one column per factor×asset ('{factor}::{ticker}'), matching the
    'pos::' / 'rpnl::' prefix convention in sleeve_backtest_log.csv. Full history
    is written (the dashboard slices to its own display start on read). This lets
    the dashboard READ these instead of re-deriving them, so changing the engine
    here can never silently desync the chart.

    범위는 sleeves.factor_monitor_scope 로 결정한다:
      traded (기본) — 실제 매매 자산만 (韓美 4종). 대시보드 '금리 자산 시그널'
                      표가 곧 주문 후보 목록이 되어 혼선이 없다.
      signal        — 시그널 유니버스 8종 (+英·日·豪).
      all           — 가격 패널 전체 12종 (+유로존).

    2026-07-22, 두 단계로 좁혔다. ① 유로존(獨·佛·伊) 제거: exclude_assets 라
    어떤 시그널에도 기여하지 않아 장식일 뿐이고, 검증된 적 없는 신호를 '매매
    후보' 옆에 띄우면 트레이더에게 근거 없는 재량 개입 압력을 준다 (유로존은
    오히려 '매매하면 손해'가 확인된 자산군). 더구나 佛·伊의 실제 정보는 아웃라이트
    방향이 아니라 對분트 스프레드(OAT-Bund, BTP-Bund = 유럽 국가신용 스트레스)에
    있는데 여기 찍히는 건 아웃라이트라 유용한 형태조차 아니다 — 유럽 스트레스가
    필요하면 스프레드 지표를 따로 만들 것.
    ② 시그널 전용(英·日·豪)도 제거 (운용자 판단): 이들은 횡단면 기준선을 만들어
    韓美 포지션을 실제로 좌우하지만, 주문할 수 없는 자산이 후보 표에 섞이는 혼선이
    그 설명 가치보다 크다. 되돌리려면 scope: signal.
    """
    scope = str((engine.cfg or {}).get('factor_monitor_scope', 'signal')).lower()
    if scope == 'all':
        ra = [c for c in engine.prices.columns if 'Comdty' in c and 'NQ' not in c]
    elif scope == 'traded':
        ra = [a for a in engine.rates_assets if a not in engine.signal_only_assets]
    else:
        ra = list(engine.rates_assets)
    # factor_monitor_extra: 시그널에도 매매에도 안 쓰지만 맥락용으로 차트에만
    # 얹는 자산 (예: 분트 = 유럽 듀레이션 벤치마크). 위 다섯 팩터는 전부 컬럼별
    # 시계열 연산(_zscore 롤링)이라 자산을 얹어도 기존 자산의 값은 변하지 않는다
    # — 횡단면 demean 은 여기가 아니라 _class_sleeves 에서 일어난다.
    for a in ((engine.cfg or {}).get('factor_monitor_extra', []) or []):
        if a in engine.prices.columns and a not in ra:
            ra.append(a)
    if not ra:
        print("⚠ No rates assets — factor signals not saved.")
        return None
    factors = {
        'trend':  engine.trend_signal(ra),
        'value':  engine.value_signal(ra),
        'carry':  engine.carry_signal(ra, 'rates'),
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


def run(start_date=None, end_date=None, target_vol=None, smooth=0.0, plot=True,
        data_start='2010-01-01', cfg_override=None, exclude_assets=None,
        save_outputs=True):
    """금리 북 백테스트. 엔진은 항상 전체 패널(data_start~)로 돌리고 결과만 start_date 부터
    잘라 낸다 — 라이브(대시보드)와 같은 워밍업이라 백테스트 초기가 콜드스타트로 왜곡되지
    않는다 (2026-10-13: 2016 에서 엔진을 새로 시작하면 2016-18 SR 0.86 vs 라이브 조건 0.52)."""
    print("📊 Loading price data...")
    loader = DataLoader()
    cfg = load_sleeve_config()
    if cfg_override:
        for k, v in cfg_override.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    if target_vol is not None:
        cfg['target_port_vol_rates'] = target_vol
    prices, yields, macro = loader.engine_inputs(cfg, start_date=data_start, asof=end_date)
    if exclude_assets:
        drop = [a for a in exclude_assets if a in prices.columns]
        prices = prices.drop(columns=drop)
        print(f"   ⛔ Excluded {len(drop)} assets: {', '.join(drop)}")

    engine = SleeveEngine(prices, config=cfg, yields=yields, macro=macro)
    # 데이터 신선도 — 오래된/누락 데이터로 시그널을 내고 있으면 크게 경고
    fr = loader.freshness_report(watch=freshness_watch(engine))
    print(f"   데이터 기준일 {fr['data_asof']} (기대 세션 {fr['expected']})")
    for msg in fr['issues']:
        print(f"   ⚠⚠ {msg}")
    for msg in fr.get('warnings', []):
        print(f"   ⚠ {msg}")
    _so = [a for a in engine.rates_assets if a in engine.signal_only_assets]
    print(f"   Rates: {len(engine.rates_assets)} 시그널 / {len(engine.rates_assets) - len(_so)} 매매 | "
          f"FX(모니터링): {len(engine.fx_assets)} | 금리 볼타겟 {engine.target_port_vol_rates:.2%} × "
          f"노출 {engine.rates_exposure_scale:g} | 캐리 {engine.carry_method}")
    if _so:
        print(f"   🎯 시그널 전용 (포지션 0): {', '.join(_so)}")

    positions = engine.finalize_positions(
        engine.compute_target_positions(),
        smooth_override=smooth if smooth and smooth > 0 else None,
    )
    if engine.book_stop_enabled:
        _w = engine.book_stop_dd_window
        print(f"   🛑 Rates book stop ON (shadow-DD half {engine.book_stop_dd_half:g}% "
              f"/ flat {engine.book_stop_dd_flat:g}%, peak = "
              f"{f'{_w}d rolling' if _w > 0 else 'all-time ⚠latch risk'})")
    if engine._dv01_state is not None:
        f = engine._dv01_state['factor']
        print(f"   📏 DV01 한도 ON (순 {engine.dv01_net_limit:,.0f} / 그로스 {engine.dv01_gross_limit:,.0f} "
              f"만원/bp @ {engine.dv01_capital:,.0f}억) — 축소 발동 {(f < 1).mean():.1%} 일")

    pnl_basis = str(cfg.get('pnl_basis', 'futures')).lower()
    if pnl_basis == 'futures' and not cfg.get('roll_adjust', False):
        print("   ⚠ pnl_basis=futures 인데 roll_adjust 가 꺼져 있음 → 롤 점프가 손익에 들어감")
    print(f"   손익 기준: {pnl_basis}  | 롤 보정: {'ON' if cfg.get('roll_adjust') else 'OFF'}")

    # 손익 회계 단일 출처 (src/sleeves/accounting.py) — 전체 기간으로 계산 후 잘라 낸다
    traded = list(positions.columns)
    net_pnl = asset_pnl(engine, positions, traded, basis=pnl_basis)
    turnover = positions.diff().abs().fillna(0.0)
    if start_date:
        keep = positions.index >= pd.to_datetime(start_date)
        positions, net_pnl, turnover = positions[keep], net_pnl[keep], turnover[keep]

    rates_cols = engine.rates_assets
    fx_cols = engine.fx_assets
    port = net_pnl.sum(axis=1)
    rates = net_pnl[rates_cols].sum(axis=1) if rates_cols else pd.Series(0.0, index=port.index)
    fx = net_pnl[fx_cols].sum(axis=1) if fx_cols else pd.Series(0.0, index=port.index)

    # ── Report ───────────────────────────────────────────────────────────
    avg_gross = positions[rates_cols].abs().sum(axis=1).mean()
    ann_turnover = turnover[rates_cols].sum(axis=1).mean() * TRADING_DAYS
    print("\n" + "=" * 64)
    print(f"  SLEEVE BACKTEST  {positions.index[0].date()} → {positions.index[-1].date()}")
    print("=" * 64)
    for name, series in [('  Rates', rates), ('  FX(모니터링)', fx)]:
        s = perf_stats(series)
        print(f"{name:<14} Sharpe {s['sharpe']:5.2f} | Vol {s['vol']:5.1%} | "
              f"AnnRet {s['ret']:6.1%} | MaxDD {s['maxdd']:6.1%}")
    print(f"\n  금리 평균 그로스 {avg_gross:.2f}x | 연 회전율 {ann_turnover:.1f}x")

    # ── Plot ───────────────────────────────────────────────────────────────
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[3, 1])
        s = perf_stats(rates)
        (1 + rates).cumprod().plot(ax=ax[0], label=f"Rates (SR {s['sharpe']:.2f})")
        ax[0].set_title(f"Sleeve Engine — rates book (vol target {engine.target_port_vol_rates:.2%} x "
                        f"exposure {engine.rates_exposure_scale:g})")
        ax[0].set_ylabel("Cumulative (1+r)"); ax[0].grid(alpha=0.3); ax[0].legend()
        positions[rates_cols].abs().sum(axis=1).plot(ax=ax[1], color='gray', lw=0.8)
        ax[1].set_ylabel("Gross lev"); ax[1].grid(alpha=0.3)
        sd = positions.index[0].strftime('%Y-%m-%d')
        out = f"sleeve_backtest_{sd}_tv10.png"
        plt.tight_layout(); plt.savefig(out, dpi=110)
        print(f"\n📊 Saved plot → {out}")

    # Save daily series (PM FACTOR_RATES · 대시보드 차트의 원천)
    out_df = pd.DataFrame({'portfolio': port, 'rates': rates, 'fx': fx})
    out_df['equity'] = (1 + port).cumprod()
    # 매매 자산만 기록 (시그널 전용은 항상 0). rpnl:: 합 = 'rates', pos:: = 목표 포지션.
    traded_cols = [a for a in rates_cols if a not in engine.signal_only_assets]
    for a in traded_cols:
        out_df[f'rpnl::{a}'] = net_pnl[a]
    for a in traded_cols:
        out_df[f'pos::{a}'] = positions[a]
    if save_outputs:
        out_df.to_csv('sleeve_backtest_log.csv')
        print("✅ Saved sleeve_backtest_log.csv")
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
