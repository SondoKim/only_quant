# -*- coding: utf-8 -*-
"""비가격 신규 정보원 4종 홀드아웃 A/B (2026-08-28).

배경: 2026-08 재검증 — 이 북의 알파는 사실상 '방향성 금리 모멘텀' 하나
(trend↔policy 상관 +0.84)이고, 가격 파생 신규 팩터는 중복 확률이 높다.
첫 비가격 시도(BEI/CPI·OIS gap·MOVE)는 홀드아웃에서 전부 기각됐다
(test_macro_factors.py / test_move_overlay.py). 이번 4종은 포지셔닝·서베이·
서프라이즈·텀프리미엄 — 남아 있던 비가격 정보원들이다.

설계 노트: 후보 슬리브는 전부 시계열(xs_neutralize 0, 매크로 슬리브 관례)
이므로 signal_only 자산(英日豪)에는 효과가 없다 → 미국(TU/TY)·한국(KE/KAA)
커버리지만 실질 의미. CESI 만 한국(CESIKRW)을 커버한다.

데이터 (data/cache/, 2026-08-28 확보):
  newinfo_macro_*.parquet — NAPMPMI(ISM, 2000+), CESIUSD/KRW 외(2003+),
      ACMTP02/05/10(2000+). S&P Global PMI(한/영/일/호)는 터미널 히스토리가
      2023-08+ 3년뿐이라 사용 불가 — 성장 슬리브는 美 ISM 단독.
  cot_ust_*.parquet — CFTC legacy futures-only, 비상업 순포지션/미결제약정,
      UST 2Y/10Y, 화요일 스탬프 (cftc.gov 공식 히스토리, 2008+).

사전등록 시그널 (재튜닝·부호반전 금지):
  B growth: 美 ISM PMI, 월말 스탬프 → 3거래일 발표랙 → −z252(Δ63d).
      PMI 상승 = 성장 상방 = 숏 듀레이션. TU/TY.
  C cot:    net%OI 화요일 스탬프 → 4거래일 랙(금요일 발표) → −z252(레벨).
      스펙 쏠림 컨트래리언 (2018 숏 극단 → 2019 랠리 류). TU←2y, TY←10y.
  D cesi:   CESI 레벨 shift(1) → −z252. 서프라이즈 상방 = 숏 듀레이션.
      TU/TY←CESIUSD, KE/KAA←CESIKRW.
  E acm:    ACM 텀프리미엄 레벨 shift(1) → +z504 (value_window 관례).
      텀프리미엄 높음 = 듀레이션 저평가 = 롱. TU←ACMTP02, TY←ACMTP10.
  공통: clip ±3(signal_clip), 슬리브 가중 1.0(균등 관례), xs_neutralize 0.

사전등록 게이트 (test_macro_factors.py 와 동일 규율):
  G1 상관: 개발표본에서 후보 시그널(매매 4자산 스택) vs 기존 각 슬리브 및
      결합 컨빅션 |ρ| < 0.5 — 새 정보원인지 확인.
  G2 개발(2012-2021 판정): 북 SR 개선 & H1/H2/T+2 모두 개선 또는 동치(−0.03).
  G3 홀드아웃(2022+ 확인 전용): ΔSR ≥ −0.03.
  전부 통과한 후보만 엔진 통합 논의. 하나라도 탈락 → 기각 기록.

Usage: python scripts/test_newinfo_factors.py
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

CACHE = Path(__file__).parent.parent / 'data' / 'cache'
DEV = ('2012-01-01', '2021-12-31')
DEV_H1 = ('2012-01-01', '2016-12-31')
DEV_H2 = ('2017-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)
TRADED = ['TU1 Comdty', 'TY1 Comdty', 'KE1 Comdty', 'KAA1 Comdty']


def window(s, a, b):
    s = s[s.index >= pd.to_datetime(a)]
    if b:
        s = s[s.index <= pd.to_datetime(b)]
    return s.dropna()


def zroll(s: pd.Series, w: int) -> pd.Series:
    mp = max(20, w // 3)
    return (s - s.rolling(w, min_periods=mp).mean()) / \
        s.rolling(w, min_periods=mp).std().replace(0.0, np.nan)


class CandEngine(SleeveEngine):
    """운용 엔진 + 후보 슬리브 1종 주입 (엔진 본체 무수정 A/B 용)."""

    def __init__(self, *a, cand_sig: pd.DataFrame = None, **kw):
        self._cand_sig = cand_sig
        super().__init__(*a, **kw)

    def _class_sleeves(self, assets, asset_class):
        out = super()._class_sleeves(assets, asset_class)
        if asset_class == 'rates' and self._cand_sig is not None:
            out['cand'] = self._cand_sig.reindex(
                index=self.prices.index, columns=assets).fillna(0.0)
        return out


def build_candidates(idx: pd.DatetimeIndex, clip: float):
    """사전등록 스펙 그대로 후보 시그널 4종 생성 (거래일 인덱스 기준)."""
    ni = sorted(CACHE.glob('newinfo_macro_*.parquet'))
    ct = sorted(CACHE.glob('cot_ust_*.parquet'))
    if not ni or not ct:
        raise SystemExit('❌ newinfo/cot 캐시 없음 — 세션 기록의 확보 스크립트로 재수집 필요')
    M = pd.read_parquet(ni[-1]).reindex(idx).ffill()
    C = pd.read_parquet(ct[-1]).reindex(idx).ffill()

    out = {}

    # B growth — 美 ISM PMI 모멘텀
    ism = M['NAPMPMI Index'].shift(3)                    # 발표랙 3거래일
    g = (-zroll(ism - ism.shift(63), 252)).clip(-clip, clip)
    out['growth'] = pd.DataFrame({'TU1 Comdty': g, 'TY1 Comdty': g})

    # C cot — 스펙 순포지션 컨트래리언
    cot = C.shift(4)                                     # 금요일 발표랙
    out['cot'] = pd.DataFrame({
        'TU1 Comdty': (-zroll(cot['net_pct_2y'], 252)).clip(-clip, clip),
        'TY1 Comdty': (-zroll(cot['net_pct_10y'], 252)).clip(-clip, clip)})

    # D cesi — 경제 서프라이즈 (美·韓)
    us = (-zroll(M['CESIUSD Index'].shift(1), 252)).clip(-clip, clip)
    kr = (-zroll(M['CESIKRW Index'].shift(1), 252)).clip(-clip, clip)
    out['cesi'] = pd.DataFrame({'TU1 Comdty': us, 'TY1 Comdty': us,
                                'KE1 Comdty': kr, 'KAA1 Comdty': kr})

    # E acm — 텀프리미엄 밸류
    out['acm'] = pd.DataFrame({
        'TU1 Comdty': zroll(M['ACMTP02 Index'].shift(1), 504).clip(-clip, clip),
        'TY1 Comdty': zroll(M['ACMTP10 Index'].shift(1), 504).clip(-clip, clip)})

    return out


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    MC = ld.load_signal_macro(start_date='2010-01-01', use_cache=True)
    cfg = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(cfg.get('costs_bps', {}) or {})}
    clip = float(cfg.get('signal_clip', 3.0))

    cands = build_candidates(PX.index, clip)

    def book(engine):
        R = engine.rates_assets
        pos = engine.finalize_positions(engine.compute_target_positions())
        rets = yield_implied_returns(engine, engine.dir_returns[list(pos.columns)]
                                     .reindex(pos.index).fillna(0.0))
        crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})

        def pnl(lag):
            turn = pos[R].diff().abs().fillna(0.0)
            held = pos[R].shift(lag).fillna(0.0)
            return (held * rets[R] - turn.mul(crate, axis=1)).sum(axis=1).dropna()
        return pnl(1), pnl(2)

    def row(tag, p1, p2, base=None):
        cells = []
        for (a, b) in (DEV, DEV_H1, DEV_H2):
            cells.append(perf_stats(window(p1, a, b))['sharpe'])
        cells.append(perf_stats(window(p2, *DEV))['sharpe'])       # dev T+2
        cells.append(perf_stats(window(p1, *HOLD))['sharpe'])      # hold
        cells.append(perf_stats(window(p2, *HOLD))['sharpe'])      # hold T+2
        dd = perf_stats(window(p1, *DEV))['maxdd']
        d = ''
        if base is not None:
            d = f'  Δdev {cells[0]-base[0]:+.2f} Δhold {cells[4]-base[4]:+.2f}'
        print(f'  {tag:10s} dev {cells[0]:5.2f} | H1 {cells[1]:5.2f} | '
              f'H2 {cells[2]:5.2f} | T+2 {cells[3]:5.2f} | devDD {dd:6.1%} | '
              f'hold {cells[4]:5.2f} | holdT+2 {cells[5]:5.2f}{d}')
        return cells

    print('=' * 100)
    print('  비가격 신규 정보원 4종 — 홀드아웃 A/B (개발 2012-2021 판정 / 2022+ 확인)')
    print('=' * 100)

    base_engine = SleeveEngine(PX, config=cfg, yields=YL, macro=MC)
    b1, b2 = book(base_engine)
    base_cells = row('A 현행', b1, b2)

    # ── G1 상관 게이트 (개발표본) ────────────────────────────────────────
    R8 = base_engine.rates_assets
    base_sleeves = base_engine._class_sleeves(R8, 'rates')
    base_conv = base_engine._combine_class(R8, 'rates')

    def stacked_corr(s_df, b_df):
        pairs = []
        for a in TRADED:
            if a not in s_df.columns or a not in b_df.columns:
                continue
            pairs.append(pd.concat([window(s_df[a], *DEV), window(b_df[a], *DEV)],
                                   axis=1, keys=['c', 'b']).dropna())
        both = pd.concat(pairs, ignore_index=True)
        return both['c'].corr(both['b'])

    print('\n  G1 상관 게이트 (매매 4자산 스택, 개발표본, |ρ|<0.5):')
    corr_pass = {}
    for name, sig in cands.items():
        s_df = sig.reindex(index=PX.index).fillna(0.0)
        rows = [(bn, stacked_corr(s_df, bsig)) for bn, bsig in base_sleeves.items()]
        rows.append(('combined', stacked_corr(s_df, base_conv)))
        mx = max(abs(v) for _, v in rows)
        corr_pass[name] = mx < 0.5
        detail = ', '.join(f'{bn} {v:+.2f}' for bn, v in rows)
        print(f'    {name:7s} max|ρ| {mx:.2f} {"통과" if mx < 0.5 else "탈락"}  ({detail})')

    # ── 후보별 A/B ──────────────────────────────────────────────────────
    print('\n  후보별 북 성과 (net, yield-implied):')
    sw = {**cfg['sleeve_weights'], 'cand': 1.0}
    for name, sig in cands.items():
        cfg2 = dict(cfg)
        cfg2['sleeve_weights'] = sw
        e = CandEngine(PX, config=cfg2, yields=YL, macro=MC, cand_sig=sig)
        p1, p2 = book(e)
        row(f'+{name}', p1, p2, base=base_cells)

    print('\n  * 판정: G1 통과 + 개발 전 열 개선(또는 −0.03 이내) + 홀드아웃 ΔSR ≥ −0.03')
    print('  * 탈락 후보의 부호 반전·창 변경 재실험 금지 (같은 표본 채굴).')


if __name__ == '__main__':
    main()
