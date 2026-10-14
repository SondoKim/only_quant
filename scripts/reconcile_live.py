# -*- coding: utf-8 -*-
"""실매매 ↔ 모델 대사 (2026-10-13). 네 가지를 날짜별로 맞춰 본다.

 ① 재작성(restatement): 그날 공표한 포지션(data/live_record.csv) vs 현재 백테스트 로그의 같은 날
    포지션. 차이 = 공표 이후 모델·데이터가 바뀌어 '과거 성과'가 다시 쓰인 크기.
 ② 라이브 페이퍼 손익: 공표 포지션 × 실현 선물수익률(롤 보정, 비용 차감) vs 백테스트 로그 손익.
    공표 원장만으로 계산하므로 재작성의 영향을 받지 않는 '정직한 라이브 성과'.
 ③ PM 환산: PM 이 그날 낸 FACTOR 목표(portfolio_management/data/source_targets.csv) ÷ 공표한
    1배 계약수 = 실효 배분 배수. 상품 간 배수가 같고 부호가 일치해야 정상.
 ④ 집행: 실보유(PM 체결 원장 trade_log.csv 누적 순체결) vs PM 전 전략 합산 목표(전일 기준).
 ⓪ 재공표: live_record 는 추가 전용 — 같은 기준일을 다시 돌려 포지션이 바뀐 날(rev ≥ 1)과 그 크기.
    ①②③ 은 날짜별 마지막 공표(rev 최대)를 쓴다.
    상품 계좌는 전략을 구분하지 않으므로 FACTOR 만의 실손익은 분리할 수 없다 — ④는 '목표대로
    집행됐는가'까지만 본다 (전략별 실손익 귀속 규칙은 PM 쪽 결정 사항).

출력: 콘솔 + data/reconcile_report.json. 데이터가 쌓이기 전(①②는 공표 2일+, ③④는 PM
source_targets 기록 이후)에는 해당 섹션이 '표본 부족'으로 표시된다.

Usage: python scripts/reconcile_live.py [--pm-dir ../portfolio_management]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECORD = ROOT / 'data' / 'live_record.csv'
LOG = ROOT / 'sleeve_backtest_log.csv'
REPORT = ROOT / 'data' / 'reconcile_report.json'
PM_KEY = {'TU1 Comdty': 'TU', 'TY1 Comdty': 'TY', 'KE1 Comdty': 'KTB3', 'KAA1 Comdty': 'KTB10'}


def revisions(rec_all):
    """같은 기준일 재공표 — 첫 공표 vs 마지막 공표 포지션 차이."""
    A = _assets(rec_all)
    g = rec_all.groupby(level=0)
    multi = [d for d, n in g.size().items() if n > 1]
    rows = []
    for d in multi:
        x = rec_all.loc[[d]]
        diff = max(abs(float(x[f'pos::{a}'].iloc[-1]) - float(x[f'pos::{a}'].iloc[0])) for a in A)
        rows.append({'date': str(d.date()), 'revs': int(len(x)), 'max_abs_dpos': round(diff, 4),
                     'first': str(x['generated'].iloc[0]), 'last': str(x['generated'].iloc[-1])})
    return {'status': 'ok', 'days_revised': len(rows), 'rows': rows}


def _assets(rec):
    return [c[5:] for c in rec.columns if c.startswith('pos::')]


def restatement(rec, log):
    A = _assets(rec)
    common = rec.index.intersection(log.index)
    if not len(common):
        return {'status': '표본 부족', 'rows': []}
    d = pd.DataFrame({a: rec.loc[common, f'pos::{a}'] - log.loc[common, f'pos::{a}'] for a in A
                      if f'pos::{a}' in log.columns})
    big = d[d.abs().max(axis=1) > 0.01]
    return {'status': 'ok', 'days': len(common), 'max_abs_diff': float(d.abs().max().max()),
            'restated_days': [str(x.date()) for x in big.index],
            'rows': big.round(3).reset_index().astype(str).to_dict('records')}


def paper_pnl(rec):
    """공표 포지션 × 다음 구간 실현 선물수익률 − 비용 (src/sleeves/accounting 규약과 동일)."""
    if len(rec) < 2:
        return {'status': '표본 부족 (공표 2일 이상 필요)'}
    import logging
    logging.disable(logging.WARNING)
    from src.data.loader import DataLoader
    from src.sleeves.sleeve_engine import SleeveEngine
    from src.sleeves.accounting import asset_pnl
    from scripts.run_sleeve_backtest import load_sleeve_config
    ld = DataLoader()
    cfg = load_sleeve_config()
    px, yl, mc = ld.engine_inputs(cfg)
    e = SleeveEngine(px, config=cfg, yields=yl, macro=mc)
    A = _assets(rec)
    idx = px.index[px.index >= rec.index[0]]
    pub = pd.DataFrame({a: rec[f'pos::{a}'] for a in A}).reindex(idx).ffill()   # 공표 사이 휴일은 유지
    q = asset_pnl(e, pub, A).sum(axis=1).iloc[1:]
    return {'status': 'ok', 'from': str(idx[0].date()), 'to': str(idx[-1].date()),
            'paper_cum_bp': round(float(q.sum() * 1e4), 2), 'days': int(len(q))}


def pm_conversion(rec, pm_dir):
    f = pm_dir / 'data' / 'source_targets.csv'
    if not f.exists():
        return {'status': f'PM source_targets.csv 없음 — PM daily_update 가 다음 실행부터 기록'}
    st = pd.read_csv(f, encoding='utf-8-sig', parse_dates=['date'])
    fac = st[st['출처'] == 'FACTOR'].pivot_table(index='date', columns='상품', values='계약', aggfunc='sum')
    out = []
    for d in rec.index.intersection(fac.index):
        for a, k in PM_KEY.items():
            one = rec.loc[d].get(f'contracts_1x::{a}')
            pmv = fac.loc[d].get(k, 0)
            if one is None or not np.isfinite(one) or abs(one) < 0.5:
                continue
            out.append({'date': str(d.date()), '상품': k, '모델_1배': round(float(one), 1), 'PM_FACTOR': int(pmv),
                        '실효배수': round(float(pmv) / float(one), 3),
                        '부호일치': bool(np.sign(pmv) == np.sign(one) or pmv == 0)})
    if not out:
        return {'status': '표본 부족 (공표일과 PM 기록일 겹침 없음)'}
    df = pd.DataFrame(out)
    spread = df.groupby('date')['실효배수'].agg(lambda s: float(s.max() - s.min()))
    return {'status': 'ok', 'rows': out, 'multiple_spread_max': float(spread.max()),
            'sign_mismatch': int((~df['부호일치']).sum())}


def execution(pm_dir):
    t = pm_dir / 'data' / 'trade_log.csv'
    f = pm_dir / 'data' / 'source_targets.csv'
    if not (t.exists() and f.exists()):
        return {'status': 'PM trade_log.csv / source_targets.csv 필요'}
    trades = pd.read_csv(t, encoding='utf-8-sig', parse_dates=['체결일'])
    st = pd.read_csv(f, encoding='utf-8-sig', parse_dates=['date'])
    tot = st.pivot_table(index='date', columns='상품', values='계약', aggfunc='sum').fillna(0)
    sgn = trades['방향'].map({'매수': 1, '매도': -1}).fillna(0)
    daily = (trades['수량'] * sgn).groupby([trades['체결일'], trades['상품']]).sum().unstack(fill_value=0)
    hold = daily.cumsum()
    out = []
    keys = [k for k in PM_KEY.values() if k in tot.columns]
    for i in range(1, len(tot.index)):
        d_prev, d = tot.index[i - 1], tot.index[i]
        h = hold[hold.index <= d].iloc[-1] if (hold.index <= d).any() else pd.Series(dtype=float)
        for k in keys:
            out.append({'date': str(d.date()), '상품': k, '목표(전일)': int(tot.loc[d_prev, k]),
                        '실보유': int(h.get(k, 0)), '차이': int(h.get(k, 0) - tot.loc[d_prev, k])})
    if not out:
        return {'status': '표본 부족 (source_targets 2일 이상 필요)'}
    df = pd.DataFrame(out)
    return {'status': 'ok', 'rows': out, 'max_abs_gap': int(df['차이'].abs().max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pm-dir', default=str(ROOT.parent / 'portfolio_management'))
    args = ap.parse_args()
    pm_dir = Path(args.pm_dir)
    if not RECORD.exists():
        print("live_record.csv 없음 — scripts/live_record.py 가 daily_run 3단계에서 쌓는다")
        return 0
    from scripts.live_record import read_record
    rec_all = read_record(latest_only=False)
    rec = read_record()                       # 날짜별 마지막 공표
    log = pd.read_csv(LOG, index_col=0, parse_dates=True) if LOG.exists() else pd.DataFrame()
    rep = {'generated': pd.Timestamp.now().isoformat(timespec='seconds'),
           'published_days': len(rec), 'first': str(rec.index[0].date()), 'last': str(rec.index[-1].date()),
           'revisions': revisions(rec_all),
           'restatement': restatement(rec, log), 'paper_pnl': paper_pnl(rec),
           'pm_conversion': pm_conversion(rec, pm_dir), 'execution': execution(pm_dir)}
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(f"실매매 대사 — 공표 {rep['published_days']}일 ({rep['first']} ~ {rep['last']})")
    v = rep['revisions']
    print(f" ⓪ 재공표: {v['days_revised']}일" + (" · " + ", ".join(f"{r['date']}({r['revs']}회, 최대 |Δpos| {r['max_abs_dpos']:.3f})"
                                                     for r in v['rows'][-5:]) if v['rows'] else ''))
    r = rep['restatement']
    print(f" ① 재작성: {r['status']}" + (f" · 최대 |Δpos| {r['max_abs_diff']:.3f} · 재작성일 {len(r['restated_days'])}일"
                                       if r['status'] == 'ok' else ''))
    p = rep['paper_pnl']
    print(f" ② 라이브 페이퍼 손익: {p['status']}" + (f" · {p['from']}~{p['to']} 누적 {p['paper_cum_bp']:+.1f}bp"
                                                if p['status'] == 'ok' else ''))
    c = rep['pm_conversion']
    print(f" ③ PM 환산: {c['status']}" + (f" · 상품간 배수 편차 최대 {c['multiple_spread_max']:.3f} · 부호 불일치 {c['sign_mismatch']}"
                                        if c['status'] == 'ok' else ''))
    x = rep['execution']
    print(f" ④ 집행: {x['status']}" + (f" · 최대 |보유 − 목표| {x['max_abs_gap']}계약" if x['status'] == 'ok' else ''))
    print(f"   → {REPORT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
