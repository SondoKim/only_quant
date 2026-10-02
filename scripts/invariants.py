# -*- coding: utf-8 -*-
"""금리 북 불변식 점검 — SR 이 아니라 '계산이 말이 되는가'를 매일 확인 (2026-10-02).

배경: 2026-10-02 감사에서 백테스트·A/B 를 모두 통과한 로직 결함이 나왔다
(롤 점프 오염, KE 캐리 ≡ US 2Y, DV01 을 명목으로 표시). 성과 검증은 '틀렸지만
돈 버는 로직'을 못 잡는다 → 결과가 아니라 중간 계산의 불변식을 직접 찍는다.

점검 항목 (각각 PASS / WARN / FAIL):
  roll     롤일 수익률 잔차 — 롤 보정이 살아 있는가 (보정 후 중앙값 <1σ, 무보정 3~8σ)
  degen    팩터 퇴화 — 서로 다른 자산의 슬리브 값이 비트 동일하거나 통째로 비었는가
           (같은 입력 시리즈를 쓰도록 설계된 쌍 — policy 의 동일 2Y 맵 — 은 제외)
  dv01     DV01 계수 상식 — |β|×1e4 (= 수정듀레이션, 년)이 테너 범위 안인가
  fresh    데이터 신선도 — DataLoader.freshness_report
  pos      최종 포지션 — 유한값, 시그널 전용 자산 0, 북스톱 스케일 ∈ {0, 0.5, 1}

사용: run_invariants(engine, loader, pos) → [(항목, 상태, 설명)]
     audit_lookahead.py ⑥ 과 daily_run.py 1c 단계가 호출한다.
"""
from itertools import combinations

import numpy as np
import pandas as pd

# 매매 자산의 기대 수정듀레이션 범위 (년). β = 명목 1 당 1bp 수익률 → ×1e4 = 듀레이션.
# (호주 YM/XM 은 100−금리 호가라 β 가 듀레이션이 아님 — 시그널 전용이라 점검 제외)
EXPECTED_DURATION = {
    'TU1 Comdty': (1.2, 2.5), 'TY1 Comdty': (4.5, 8.0),
    'KE1 Comdty': (2.0, 3.5), 'KAA1 Comdty': (6.0, 10.0),
}
ROLL_Z_FAIL = 2.5         # 롤일 |잔차|/σ 중앙값이 이보다 크면 롤 점프가 시그널에 들어가는 중
DEGEN_WINDOW = 252


def _yield_beta(engine, a, lookback=250):
    yt = engine.tradeable_yield_map.get(a)
    if engine.yields is None or yt is None or yt not in engine.yields.columns:
        return None, None
    dy = engine.yields[yt].diff() * 100.0
    r = engine.dir_returns[a]
    b = (r.rolling(lookback, min_periods=60).cov(dy)
         / dy.rolling(lookback, min_periods=60).var()).shift(1)
    return b, dy


def check_roll(engine, loader, assets, years=3):
    if not engine.cfg.get('roll_adjust', False):
        return [('roll', 'FAIL', 'roll_adjust OFF — 무보정 연속선물(롤 점프 포함)로 시그널 계산 중')]
    info = loader.load_roll_info()
    if info is None or info.empty:
        return [('roll', 'FAIL', '롤 정보 없음 — 보정이 적용되지 않음')]
    out = []
    start = engine.prices.index[-1] - pd.DateOffset(years=years)
    for a in assets:
        gc = f'gen::{a}'
        b, dy = _yield_beta(engine, a)
        if gc not in info.columns or b is None:
            out.append(('roll', 'WARN', f'{a}: 롤 정보/일드 없음 — 점검 불가'))
            continue
        g = info[gc].reindex(engine.prices.index).ffill()
        roll = (g != g.shift()) & g.notna() & g.shift().notna()
        res = engine.dir_returns[a] - b * dy
        win = res.index >= start
        sd = res[win & ~roll].std()
        z = (res[win & roll].abs() / sd).dropna()
        if z.empty or not np.isfinite(sd) or sd == 0:
            out.append(('roll', 'WARN', f'{a}: 최근 {years}년 롤일 없음/표본 부족'))
            continue
        st = 'FAIL' if z.median() > ROLL_Z_FAIL else 'PASS'
        out.append(('roll', st, f'{a}: 롤일 {len(z)}회 잔차 중앙값 {z.median():.2f}σ '
                                f'(최대 {z.max():.1f}σ, 기준 ≤{ROLL_Z_FAIL})'))
    return out


def check_degenerate(engine, traded):
    w = engine.sleeve_weights
    R = engine.rates_assets
    sleeves = {}
    if w.get('trend'):
        sleeves['trend'] = engine.trend_signal(R)
    if w.get('value'):
        sleeves['value'] = engine.value_signal(R, 'rates')
    if w.get('carry'):
        sleeves['carry'] = engine.carry_signal(R, 'rates')
    if w.get('policy'):
        sleeves['policy'] = engine.policy_signal(R)
    out = []
    for name, sig in sleeves.items():
        s = sig.reindex(columns=R).tail(DEGEN_WINDOW).astype(float)
        bad = []
        for a in traded:                       # 매매 자산 신호가 통째로 비었나
            col = s.get(a)
            if col is None or col.isna().all() or (col.fillna(0) == 0).all():
                bad.append(f'{a} 신호 없음')
        for a, b in combinations(R, 2):        # 서로 다른 자산이 비트 동일한가
            if name == 'policy' and engine.policy_rate_map.get(a) == engine.policy_rate_map.get(b):
                continue                       # 같은 2Y 를 쓰도록 설계된 쌍
            x, y = s[a], s[b]
            both = x.notna() & y.notna()
            if both.sum() > 20 and (x[both] - y[both]).abs().max() < 1e-9:
                bad.append(f'{a} ≡ {b}')
        out.append(('degen', 'FAIL' if bad else 'PASS',
                    f'{name}: ' + ('; '.join(bad) if bad else f'최근 {DEGEN_WINDOW}일 퇴화 없음')))
    return out


def check_dv01(engine, traded):
    out = []
    for a in traded:
        b, dy = _yield_beta(engine, a)
        if b is None:
            out.append(('dv01', 'WARN', f'{a}: 일드 없음 — DV01 환산 불가'))
            continue
        dur = abs(b.iloc[-1]) * 1e4
        lo, hi = EXPECTED_DURATION.get(a, (0.5, 30.0))
        st = 'PASS' if lo <= dur <= hi else 'FAIL'
        out.append(('dv01', st, f'{a}: |β|×1e4 = {dur:.2f}년 (기대 {lo}~{hi})'))
    return out


def check_positions(engine, pos):
    out = []
    last = pos.iloc[-1]
    nonfinite = [a for a in pos.columns if not np.isfinite(last[a])]
    so = [a for a in engine.signal_only_assets if a in pos.columns and abs(last[a]) > 1e-12]
    msgs = []
    if nonfinite:
        msgs.append(f'비유한 포지션 {nonfinite}')
    if so:
        msgs.append(f'시그널 전용인데 포지션 ≠ 0: {so}')
    st = engine.book_stop_status()
    if st and st['scale'] not in (0.0, 0.5, 1.0):
        msgs.append(f"북스톱 스케일 이상 {st['scale']}")
    out.append(('pos', 'FAIL' if msgs else 'PASS',
                '; '.join(msgs) if msgs else
                f'{pos.index[-1].date()} 포지션 유한 · 시그널 전용 0 · 스톱 {st["scale"] if st else "-"}'))
    return out


def run_invariants(engine, loader, pos, watch=None):
    traded = [a for a in engine.rates_assets if a not in engine.signal_only_assets]
    res = []
    res += check_roll(engine, loader, traded)
    res += check_degenerate(engine, traded)
    res += check_dv01(engine, traded)
    fr = loader.freshness_report(watch=watch)
    res += [('fresh', 'FAIL', m) for m in fr['issues']] or \
           [('fresh', 'PASS', f"데이터 기준일 {fr['data_asof']} (기대 {fr['expected']})")]
    res += check_positions(engine, pos)
    return res


def print_invariants(res, indent='   '):
    icon = {'PASS': '✅', 'WARN': '⚠ ', 'FAIL': '❌'}
    for item, st, msg in res:
        print(f"{indent}{icon.get(st, st)} [{item:<5}] {msg}")
    n_fail = sum(1 for _, s, _ in res if s == 'FAIL')
    n_warn = sum(1 for _, s, _ in res if s == 'WARN')
    print(f"{indent}→ FAIL {n_fail} · WARN {n_warn} · PASS {len(res) - n_fail - n_warn}")
    return n_fail
