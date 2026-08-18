# -*- coding: utf-8 -*-
"""MOVE(채권 내재변동성) 활용 실험 (2026-08-14).

데이터: signal_macro 캐시의 'MOVE Index' (2010+, 2026-08-14 확보). 엔진은
건드리지 않고 스크립트에서 오버레이/미니북으로 검증 — 채택 시에만 엔진 통합.

⚠ 사전 경고: 과거 A/B 에서 빠른 가격/변동성 게이트는 전부 SR 을 깎았고
(2026-06-11, 북스톱만 생존), Hurst 게이트도 2026-08-14 무효 판정으로 제거됐다.
MOVE 는 '내재' 변동성이라 실현볼 EWMA(halflife 33d)보다 선행한다는 것이
유일한 신규 정보 가설이다.

사전등록 격자 (사후 세분화·부호 뒤집기 금지):
  V1a 리스크 오버레이(선형): scale = clip(med252(MOVE)/MOVE, 0.5, 1.0),
      전일값(shift 1)을 금리 최종 포지션에 곱한다. 내재볼이 1년 중앙값보다
      높을 때만 감축, 레버업 없음.
  V1b 리스크 오버레이(제곱): scale = clip((med/MOVE)^2, 0.25, 1.0) — 강한 감축.
  V2  알파 미니북: z252(MOVE) 레벨, 클립 ±2, shift 1 → 美 듀레이션(TU/TY)
      롱/숏. 경제 근거(사전등록): 내재볼 고평가 = 채권 리스크 프리미엄 확대
      → 이후 듀레이션 보유 보상 (bond risk premia 문헌의 vol-premium 채널).
      高 MOVE → 롱 듀레이션.

판정 (사전등록):
  V1: 개발표본에서 SR 개선 + MaxDD 개선, 홀드아웃 ΔSR ≥ −0.03 일 때만 채택.
  V2: 개발 standalone SR > 0.3, 홀드아웃 ≥ 0, 본북 상관 < 0.3, 20% 리스크
      블렌드가 양 구간 개선 — 전부 만족 시에만 엔진 통합 논의.

Usage: python scripts/test_move_overlay.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader                            # noqa: E402
from src.data.preprocessor import DataPreprocessor                # noqa: E402
from src.sleeves.sleeve_engine import SleeveEngine, TRADING_DAYS  # noqa: E402
from scripts.run_sleeve_backtest import (                         # noqa: E402
    load_sleeve_config, perf_stats, cost_bps_for, DEFAULT_COSTS_BPS)
from scripts.test_curve_blocks import yield_implied_returns       # noqa: E402

DEV = ('2012-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)
US = ['TU1 Comdty', 'TY1 Comdty']


def window(s, a, b):
    s = s[s.index >= pd.to_datetime(a)]
    if b:
        s = s[s.index <= pd.to_datetime(b)]
    return s.dropna()


def line(name, s, extra=""):
    st = perf_stats(s)
    print(f"  {name:<26} SR {st['sharpe']:5.2f} | AnnRet {st['ret']:6.2%} | "
          f"MaxDD {st['maxdd']:6.1%}{extra}")


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    MC = ld.load_signal_macro(start_date='2010-01-01', use_cache=True)
    cfg = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(cfg.get('costs_bps', {}) or {})}

    e = SleeveEngine(PX, config=cfg, yields=YL, macro=MC)
    R = e.rates_assets
    pos = e.finalize_positions(e.compute_target_positions())
    rets = yield_implied_returns(e, e.dir_returns[list(pos.columns)]
                                 .reindex(pos.index).fillna(0.0))
    crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})

    move = MC['MOVE Index'].reindex(pos.index).ffill()

    def book_pnl(p, lag=1):
        turn = p[R].diff().abs().fillna(0.0)
        held = p[R].shift(lag).fillna(0.0)
        return (held * rets[R] - turn.mul(crate, axis=1)).sum(axis=1).dropna()

    base = book_pnl(pos)
    base_t2 = book_pnl(pos, lag=2)

    # ── 진단: MOVE 의 예측력 (개발표본만, 참고용) ─────────────────────
    z = ((move - move.rolling(252, min_periods=60).mean())
         / move.rolling(252, min_periods=60).std()).clip(-2, 2)
    us_ret = e.dir_returns[US].mean(axis=1)
    print("=" * 74)
    print("  진단 (개발표본 2012-2021): z252(MOVE) vs 美 듀레이션 향후 수익률 상관")
    print("=" * 74)
    for h in (1, 5, 21, 63):
        fwd = us_ret.rolling(h).sum().shift(-h)   # t 시점 기준 t+1..t+h 합
        c = window(z, *DEV).corr(window(fwd, *DEV))
        print(f"    h={h:>3}d: corr {c:+.3f}")
    c_lvl = window(move, *DEV).corr(window(
        e._realized_vol()[US].mean(axis=1), *DEV))
    print(f"    (레벨 sanity: MOVE vs US 실현볼 상관 {c_lvl:+.2f} — 같은 정보인지)")

    # ── V1: 리스크 오버레이 ──────────────────────────────────────────
    med = move.rolling(252, min_periods=60).median()
    ratio = (med / move)
    scales = {
        'V1a 선형 (floor .5)':  ratio.clip(0.5, 1.0).shift(1).fillna(1.0),
        'V1b 제곱 (floor .25)': (ratio ** 2).clip(0.25, 1.0).shift(1).fillna(1.0),
    }
    for tag, (a, b) in [('개발 2012-2021', DEV), ('홀드아웃 2022+', HOLD)]:
        print("\n" + "=" * 74)
        print(f"  V1 리스크 오버레이 — {tag}")
        print("=" * 74)
        line('베이스', window(base, a, b),
             f" | T+2 {perf_stats(window(base_t2, a, b))['sharpe']:.2f}")
        for nm, sc in scales.items():
            p2 = pos.copy()
            p2[R] = p2[R].mul(sc, axis=0)
            s1 = window(book_pnl(p2), a, b)
            s2 = window(book_pnl(p2, lag=2), a, b)
            avg_sc = float(window(sc, a, b).mean())
            line(nm, s1, f" | T+2 {perf_stats(s2)['sharpe']:.2f} | 평균스케일 {avg_sc:.2f}")

    # ── V2: 알파 미니북 (高 MOVE → 롱 美 듀레이션) ───────────────────
    inv_vol = (e.target_asset_vol / e._realized_vol()[US]).clip(upper=15.0)
    mini = pd.DataFrame(0.0, index=pos.index, columns=R)
    for a2 in US:
        mini[a2] = (z.shift(1) * inv_vol[a2] * 0.5).fillna(0.0)
    mini_pnl = book_pnl(mini)

    print("\n" + "=" * 74)
    print("  V2 알파 미니북 (z252(MOVE) → 美 듀레이션, 사전등록 부호: 高→롱)")
    print("=" * 74)
    for tag, (a, b) in [('개발 2012-2021', DEV), ('홀드아웃 2022+', HOLD)]:
        s = window(mini_pnl, a, b)
        bb = window(base, a, b)
        corr = s.corr(bb.reindex(s.index))
        line(f'{tag} standalone', s, f" | 본북 상관 {corr:+.2f}")
        # 20% 리스크 블렌드 (개발표본 변동성으로 정규화 — 상수 스케일)
        sb = window(base, *DEV).std()
        sm = window(mini_pnl, *DEV).std()
        blend = 0.8 * bb / sb + 0.2 * s.reindex(bb.index).fillna(0.0) / sm
        line(f'{tag} 80/20 블렌드(정규화)', blend.dropna(),
             f" | 베이스(정규화) SR {perf_stats(bb / sb)['sharpe']:.2f}")

    print("\n  * 판정 기준은 파일 docstring 의 사전등록 항목 참조.")
    print("  * V2 부호를 뒤집어 보고 싶어도 금지 — 같은 표본 부호 채굴.")


if __name__ == '__main__':
    main()
