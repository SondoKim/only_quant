"""
Global Macro Rates Book — 라이브 시그널 CLI (SleeveEngine)

2026-09-11 FX 전략 공장 폐기(git tag `pre-fx-factory-retire`) 이후 이 파일은
금리 북 슬리브 엔진의 얇은 CLI 다. 전략 탐색(discover)·월별 갱신(update)은
팩토리와 함께 사라졌고, 시그널은 scripts/strategy_dashboard.run() 과 같은
코드 경로를 쓴다 (콘솔 표 = 대시보드 표 = 주문 후보).

    python main.py                       # = --mode signals
    python main.py --mode signals        # 금리 주문 후보 + 자동 해석 + FX 모니터링
    python main.py --mode summary        # 엔진 설정·유니버스·최신 목표 포지션 요약
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:  # 콘솔 한글/기호 깨짐 방지
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def _summary() -> int:
    import yaml
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.sleeves.sleeve_engine import SleeveEngine

    cfg_path = Path(__file__).parent / 'config' / 'indicators.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = (yaml.safe_load(f) or {}).get('sleeves', {}) or {}
    loader = DataLoader()
    px = DataPreprocessor(loader.load_data(start_date='2010-01-01', use_cache=True)).clean().get_data()
    yields = loader.load_signal_yields(start_date='2010-01-01', use_cache=True)
    e = SleeveEngine(px, config=cfg, yields=yields)
    pos = e.finalize_positions(e.compute_target_positions())
    traded = [a for a in e.rates_assets if a not in e.signal_only_assets]
    print("=" * 64)
    print("  금리 북 (SleeveEngine) 요약")
    print("=" * 64)
    print(f"  데이터: {px.index[0].date()} → {px.index[-1].date()}  ({len(px)} 일)")
    print(f"  시그널 유니버스 {len(e.rates_assets)}종: {', '.join(e.rates_assets)}")
    print(f"  매매 {len(traded)}종: {', '.join(traded)}")
    print(f"  시그널 전용: {', '.join(sorted(e.signal_only_assets)) or '-'}")
    print(f"  제외: {', '.join(cfg.get('exclude_assets', []) or []) or '-'}")
    print(f"  슬리브 가중: { {k: v for k, v in e.sleeve_weights.items() if v} }")
    print(f"  xs 중립화: {e.xs_neutralize}")
    print(f"  볼타겟: {e.vol_target_mode} (금리 {e.target_port_vol_rates:.1%}) · "
          f"position_smooth {e.position_smooth} · exposure_scale {e.rates_exposure_scale}")
    print(f"  북스톱: {'ON' if e.book_stop_enabled else 'OFF'} "
          f"(half {e.book_stop_dd_half:.0f}% / flat {e.book_stop_dd_flat:.0f}% / "
          f"peak {e.book_stop_dd_window}d) → 현재 {e.book_stop_status()}")
    last = pos.iloc[-1]
    print(f"  최신 목표 포지션 ({pos.index[-1].date()}):")
    for a in traded:
        print(f"    {a:<14} {float(last.get(a, 0.0)):+.3f}")
    print(f"  FX (모니터링 전용, 매매 안 함): "
          + ", ".join(f"{a} {float(last.get(a, 0.0)):+.2f}" for a in e.fx_assets))
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description='Global Macro Rates Book — SleeveEngine CLI')
    p.add_argument('--mode', choices=['signals', 'summary'], default='signals')
    p.add_argument('--asset', default=None, help='특정 자산만 (티커 일부, 예: TU1)')
    p.add_argument('--per-unit', type=float, default=1252.0,
                   help="금리 '포지션 1.0 = N만원' 환산 계수 (기본 1252)")
    p.add_argument('--delta-budget', type=float, default=5000.0, help='순델타 한도 (만원)')
    p.add_argument('--gross-budget', type=float, default=8000.0, help='그로스 한도 (만원)')
    args = p.parse_args()

    if args.mode == 'summary':
        sys.exit(_summary())
    from scripts.strategy_dashboard import run
    sys.exit(run(asset=args.asset, per_unit=args.per_unit,
                 delta_budget=args.delta_budget, gross_budget=args.gross_budget,
                 html_out=False))


if __name__ == '__main__':
    main()
