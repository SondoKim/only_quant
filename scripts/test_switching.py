# -*- coding: utf-8 -*-
"""금리 북 '기민한 스위칭' 후보 A/B (2026-09-03).

배경: 2026 YTD 韓美 4종 100% 숏, 부호 전환 0회 (2025 는 북스톱 플랫 83% +
나머지 숏). 방향은 trend+policy(상관 0.84) 하나가 결정하고 value/carry 는
xs 중립(횡단면 합 0)이라 방향에 기여하지 않는다. 게다가 두 모멘텀의 정규화가
'h일 변화량의 252d 롤링 평균 차감 z-score' 라 1년 같은 방향이면 롤링 평균이
현재값을 따라잡아 신호가 0 으로 퇴색한다 (2026-07~08 KR policy = 0.00).
사용자 요청: 더 기민한 순매수/순매도 전환 + 우상향.

후보 (전부 config 토글, 기본값 = 기존 동작 비트 동일 — check_identity 참조):
  B/C value xs_neutralize 1.0→0.5/0.0 — 방향성 밸류(싸지면 롱) 목소리 추가.
      기존 A/B(test_xs_neutralize/test_directionality)는 trend/policy 를 '더'
      중립화하는 쪽만 봤고 value 를 '덜' 중립화하는 쪽은 미검증.
  D   policy_periods [63,126,252] — 정책 모멘텀 다중 호라이즌 (trend 와 동형).
  E   momentum_norm 'vol' — 평균 차감 없는 표준 TSMOM 정규화 (신호 퇴색 제거).
  F/G trend_method 'ewmac' — Baz et al. 3속도 EWMA 크로스오버 (z / 반응함수).
  H   F + D 결합.
  I   trend_horizons [63,126,252] — 참조 (과거 두 차례 기각, 당시 리버전 ON).
  J   conviction_response — 결합 컨빅션에 반응함수 x·exp(−x²/4)/0.89: 속도를
      올리는 대신 |x|>√2 과열 구간에서 감축(뒤집지 않음). 비-속도 대안.

프로토콜: test_sleeve_composition.py 와 동일 — 신호는 전 히스토리로 계산,
개발 2012-2021 판정 / 홀드아웃 2022+ 확인, yield-implied 귀속, net of costs,
T+2·시간대정직 열. 게이트: 개발 전 열 ≥ 베이스 −0.03 → 홀드아웃 열람 →
hold ΔSR ≥ −0.03. ⚠ 홀드아웃은 이미 여러 라운드 열람됨 — 아슬아슬한 통과
채택 금지. 정보용 추가 열: 부호전환/yr(개발), 63d 방향 적중률(개발),
2026 YTD SR, 2025-26 숏 비율.

⚠ 토글은 금리/FX 북 공유이나 vol_target_mode=separate 라 금리 열은 오염되지
않고, 운용 FX 북은 팩토리라 무관. FX xs 는 현행값에 고정한다.

Usage: python scripts/test_switching.py
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
from scripts.test_curve_blocks import yield_implied_returns, merged  # noqa: E402

DEV = ('2012-01-01', '2021-12-31')
HOLD = ('2022-01-01', None)
EARLY = ['JB1 Comdty', 'KE1 Comdty', 'KAA1 Comdty', 'G 1 Comdty',
         'YM1 Comdty', 'XM1 Comdty']
TOL = 0.03

CASES = [
    ('A. 현행',                      {}),
    ('B. value xs 0.5',              {'xs_neutralize': {'value': 0.5}}),
    ('C. value xs 0.0',              {'xs_neutralize': {'value': 0.0}}),
    ('D. policy [63,126,252]',       {'policy_periods': [63, 126, 252]}),
    ('E. mom_norm vol',              {'momentum_norm': 'vol'}),
    ('F. trend ewmac (z)',           {'trend_method': 'ewmac'}),
    ('G. trend ewmac (resp)',        {'trend_method': 'ewmac', 'ewmac_response': True}),
    ('H. ewmac + policy multi',      {'trend_method': 'ewmac',
                                      'policy_periods': [63, 126, 252]}),
    ('I. trend [63,126,252] 참조',   {'trend_horizons': [63, 126, 252]}),
    # J: 속도가 아니라 '과열 감축' — 결합 컨빅션에 반응함수 (뒤집지 않음).
    ('J. conviction response',       {'conviction_response': True}),
]


def window(s, a, b):
    s = s[s.index >= pd.to_datetime(a)]
    if b:
        s = s[s.index <= pd.to_datetime(b)]
    return s.dropna()


def main():
    ld = DataLoader()
    PX = DataPreprocessor(ld.load_data(start_date='2010-01-01',
                                       use_cache=True)).clean().get_data()
    YL = ld.load_signal_yields(start_date='2010-01-01', use_cache=True)
    BASE = load_sleeve_config()
    COSTS = {**DEFAULT_COSTS_BPS, **(BASE.get('costs_bps', {}) or {})}
    FX_PIN = dict(BASE.get('xs_neutralize') or
                  {'trend': 0.0, 'value': 1.0, 'carry': 1.0})

    def run(ov):
        cfg = merged(BASE, ov)
        cfg['xs_neutralize_fx'] = FX_PIN
        e = SleeveEngine(PX, config=cfg, yields=YL)
        R = e.rates_assets
        T = [a for a in R if a not in e.signal_only_assets]
        pos = e.finalize_positions(e.compute_target_positions())
        dirr = e.dir_returns[R + e.fx_assets].reindex(pos.index).fillna(0.0)
        rets = yield_implied_returns(e, dirr)
        crate = pd.Series({a: cost_bps_for(a, COSTS) / 10000.0 for a in R})
        turn = pos[R].diff().abs().fillna(0.0)

        def pnl(lag_map):
            held = pd.DataFrame({a: pos[a].shift(lag_map.get(a, 1))
                                 for a in R}).fillna(0.0)
            return (held * rets[R]
                    - turn.mul(crate, axis=1)).sum(axis=1).dropna()

        s1 = pnl({a: 1 for a in R})
        s2 = pnl({a: 2 for a in R})
        stz = pnl({a: (2 if a in EARLY else 1) for a in R})
        d1 = window(s1, *DEV)
        mid = len(d1) // 2

        # 스위칭 지표 (개발표본, 매매 자산 평균)
        flips, hits = [], []
        for a in T:
            p = window(pos[a], *DEV)
            sg = np.sign(p)
            nz = sg[sg != 0]
            flips.append(((nz != nz.shift(1)).sum() - 1) / (len(p) / TRADING_DAYS))
            fwd = rets[a].rolling(63).sum().shift(-63).reindex(p.index)
            ok = (sg != 0) & fwd.notna()
            hits.append((np.sign(fwd[ok]) == sg[ok]).mean())
        y26 = window(s1, '2026-01-01', None)
        sh = pd.concat([window(pos[a], '2025-01-01', None) for a in T])
        return {
            'dev': perf_stats(d1)['sharpe'],
            'h1': perf_stats(d1.iloc[:mid])['sharpe'],
            'h2': perf_stats(d1.iloc[mid:])['sharpe'],
            't2': perf_stats(window(s2, *DEV))['sharpe'],
            'tz': perf_stats(window(stz, *DEV))['sharpe'],
            'dd': perf_stats(d1)['maxdd'],
            's16': perf_stats(window(s1, '2016-01-01', None))['sharpe'],
            'hold': perf_stats(window(s1, *HOLD))['sharpe'],
            'hdd': perf_stats(window(s1, *HOLD))['maxdd'],
            'turn': float(window(turn.sum(axis=1), *DEV).mean() * TRADING_DAYS),
            'flips': float(np.mean(flips)),
            'hit63': float(np.mean(hits)),
            'sr26': perf_stats(y26)['sharpe'],
            'ret26': float(y26.sum()),
            'short2526': float((sh < 0).mean()),
        }

    GATE = ['dev', 'h1', 'h2', 't2', 'tz']

    def dev_pass(r, base):
        return all(r[c] >= base[c] - TOL for c in GATE)

    hdr = (f"{'구성':<28}{'devSR':>7}{'H1':>6}{'H2':>6}{'T+2':>6}{'정직':>6}"
           f"{'devDD':>7}{'2016+':>7}{'hold':>6}{'hDD':>7}{'회전':>6}"
           f"{'전환/yr':>8}{'hit63':>7}{'SR26':>6}{'ret26':>7}{'숏%':>5}")
    print('=' * len(hdr))
    print('  금리 북 스위칭 후보 A/B — 개발 2012-2021 판정 / 홀드아웃 2022+ 확인')
    print('=' * len(hdr))
    print(hdr)
    base = None
    passers = []
    for name, ov in CASES:
        r = run(ov)
        if base is None:
            base = r
            ok = True
        else:
            ok = dev_pass(r, base)
        hold = f"{r['hold']:>6.2f}{r['hdd']:>7.1%}" if ok else f"{'—':>6}{'—':>7}"
        print(f"{('✓ ' if ok and base is not r else '  ') + name:<28}"
              f"{r['dev']:>7.2f}{r['h1']:>6.2f}{r['h2']:>6.2f}{r['t2']:>6.2f}"
              f"{r['tz']:>6.2f}{r['dd']:>7.1%}{r['s16']:>7.2f}{hold}"
              f"{r['turn']:>5.0f}x{r['flips']:>8.1f}{r['hit63']:>7.1%}"
              f"{r['sr26']:>6.2f}{r['ret26']:>7.1%}{r['short2526']:>5.0%}")
        if ok and base is not r:
            passers.append((name, r))

    print()
    print('  게이트: 개발 전 열(devSR/H1/H2/T+2/정직) ≥ 베이스 −0.03 → 홀드아웃 열람'
          ' → hold ΔSR ≥ −0.03. 탈락 구성의 hold 는 비공개(—).')
    print('  전환/yr·hit63 = 개발표본 매매 4종 평균; SR26/ret26/숏% 는 정보용'
          ' (2026 YTD, 2025-26 매매자산 숏 일수 비율).')
    if not passers:
        print('  → 개발 게이트 통과 구성 없음: 현행 유지.')
    else:
        for name, r in passers:
            v = ('홀드아웃 통과' if r['hold'] >= base['hold'] - TOL else '홀드아웃 탈락')
            print(f"  → {name}: dev {r['dev']:.2f} (베이스 {base['dev']:.2f}), "
                  f"hold {r['hold']:.2f} (베이스 {base['hold']:.2f}) — {v}")
        print('  ⚠ 홀드아웃 재사용 상태 — 아슬아슬한 통과는 채택하지 않는다.')


if __name__ == '__main__':
    main()
