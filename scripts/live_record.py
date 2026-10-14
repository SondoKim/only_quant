# -*- coding: utf-8 -*-
"""라이브 기록 — 그날 '공표한' 금리 북 목표를 data/live_record.csv 에 추가 (2026-10-13).

왜 필요한가: sleeve_backtest_log.csv 는 매 실행마다 전체 이력을 현재 모델·현재 데이터로 다시
쓴다 (2026-10-13 감사로 과거 SR 이 1.24 → 0.72 로 바뀐 것처럼). 그래서 '그날 실제로 무엇을
권고했는가'는 어디에도 남지 않았다. 이 파일은 날짜별로 한 행씩 쌓고 과거 행을 고치지 않는다 —
실매매 대사(scripts/reconcile_live.py)와 사후 검증의 기준이다.

행 = 데이터 기준일 T 의 공표 1회. 추가만 하고 어떤 행도 고치거나 지우지 않는다 (2026-10-14: 예전엔 같은
T 재실행이 그 행을 덮어써 '처음 공표한 값'이 사라졌다 — 10-12 행이 실제 집행과 다른 새 코드 값으로 바뀐 예).
같은 T 를 다시 돌려 포지션이 바뀌면 rev 를 올려 새 행으로 쌓는다 (포지션·DV01 이 같으면 기록 생략).
  rev(같은 T 의 공표 순번, 0 부터) · generated · git(커밋, dirty 표시) · cfg_hash(운용 설정 지문) · data_asof
  checks_fail          daily_run 1c 점검 FAIL 건수 (같은 기준일 점검이 없으면 공란)
  pos::<자산>          목표 포지션 (배정자본 대비 명목 배수, T 종가 결정 → T+1 보유)
  dv01::<자산>         DV01 (만원/bp, 배정자본 기준, + = 듀레이션 롱)
  contracts_1x::<자산> 배정자본 1배 기준 목표 계약수 (PM 배분 배수 적용 전, 반올림 전)
  price::<자산>        근월물 종가 (계약 환산에 쓴 값), usdkrw
  net_dv01 · gross_dv01 · stop_scale · dv01_factor

Usage: python scripts/live_record.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader                       # noqa: E402
from src.sleeves.sleeve_engine import SleeveEngine           # noqa: E402
from scripts.run_sleeve_backtest import load_sleeve_config   # noqa: E402

RECORD = ROOT / 'data' / 'live_record.csv'
# 계약 1포인트 가치 (PM positions._POINT_VALUE 와 동일) · 통화
POINT_VALUE = {'TU1 Comdty': (2000.0, 'USD'), 'TY1 Comdty': (1000.0, 'USD'),
               'KE1 Comdty': (1_000_000.0, 'KRW'), 'KAA1 Comdty': (1_000_000.0, 'KRW')}


def _git() -> str:
    try:
        sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'config', 'scripts'], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ('+dirty' if dirty else '')
    except Exception:
        return 'unknown'


def build_row() -> pd.DataFrame:
    import logging
    logging.disable(logging.WARNING)
    ld = DataLoader()
    cfg = load_sleeve_config()
    px, yl, mc = ld.engine_inputs(cfg)
    e = SleeveEngine(px, config=cfg, yields=yl, macro=mc)
    pos = e.finalize_positions(e.compute_target_positions())
    T = pos.index[-1]
    traded = [a for a in e.rates_assets if a not in e.signal_only_assets]
    cap = float(cfg.get('capital_억', 500.0))
    beta = e.dv01_betas(traded)
    usdkrw = float(px['KRW Curncy'].iloc[-1]) if 'KRW Curncy' in px.columns else float('nan')
    row = {'rev': 0, 'generated': pd.Timestamp.now().isoformat(timespec='seconds'), 'git': _git(),
           'cfg_hash': hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10],
           'data_asof': str(T.date()), 'capital_억': cap, 'usdkrw': usdkrw}
    net = gross = 0.0
    for a in traded:
        p = float(pos[a].iloc[-1])
        b = float(beta[a].dropna().iloc[-1]) if a in beta.columns and beta[a].notna().any() else float('nan')
        dv = p * b * cap * 1e4
        net += dv
        gross += abs(dv)
        price = float(px[a].iloc[-1])            # 롤 보정 후에도 최신 값 = 실제 근월물 종가
        pv, ccy = POINT_VALUE.get(a, (float('nan'), 'KRW'))
        fx = 1.0 if ccy == 'KRW' else usdkrw
        row.update({f'pos::{a}': round(p, 6), f'dv01::{a}': round(dv, 1),
                    f'contracts_1x::{a}': round(p * cap * 1e8 / (price * pv * fx), 2),
                    f'price::{a}': price})
    try:   # daily_run 1c 점검 결과 (같은 데이터 기준일일 때만)
        ck = json.loads((ROOT / 'data' / 'daily_checks.json').read_text(encoding='utf-8'))
        row['checks_fail'] = len(ck.get('fail') or []) if str(ck.get('data_asof')) == str(T.date()) else None
    except Exception:
        row['checks_fail'] = None
    st = e.book_stop_status() or {}
    row.update({'net_dv01': round(net, 1), 'gross_dv01': round(gross, 1),
                'stop_scale': st.get('scale'), 'stop_dd': st.get('dd'),
                'dv01_factor': (float(e._dv01_state['factor'].iloc[-1]) if e._dv01_state else None)})
    return pd.DataFrame([row], index=pd.Index([T.date()], name='date'))


def read_record(latest_only: bool = True) -> pd.DataFrame:
    """공표 원장 읽기. latest_only = 같은 날짜의 마지막 공표(rev 최대)만 (대사·페이퍼 손익 기본)."""
    rec = pd.read_csv(RECORD, index_col='date', parse_dates=True)
    if 'rev' not in rec.columns:            # 2026-10-14 이전 형식 (날짜당 1행)
        rec.insert(0, 'rev', 0)
    rec = rec.reset_index().sort_values(['date', 'rev']).set_index('date')
    return rec[~rec.index.duplicated(keep='last')] if latest_only else rec


def main() -> int:
    row = build_row()
    T = pd.Timestamp(row.index[0])
    row.index = pd.DatetimeIndex([T], name='date')
    if RECORD.exists():
        old = read_record(latest_only=False)
        later = old.index[old.index > T]
        if len(later):
            print(f"⚠ live_record 에 {T.date()} 이후 행이 있음 ({later[-1].date()}) — 데이터가 뒤로 간 실행, 기록하지 않음")
            return 0
        same = old[old.index == T]
        if len(same):
            last = same.iloc[-1]
            cols = [c for c in row.columns if c.startswith(('pos::', 'dv01::'))]
            if all(c in last.index and abs(float(last[c]) - float(row[c].iloc[0])) < 1e-6 for c in cols):
                print(f"✅ live_record {T.date()} rev {int(last['rev'])} 와 포지션·DV01 동일 — 기록 생략")
                return 0
            row['rev'] = int(same['rev'].max()) + 1
        out = pd.concat([old, row])
    else:
        out = row
    out.index.name = 'date'
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    out = out[['rev'] + [c for c in out.columns if c != 'rev']]
    out.to_csv(RECORD, date_format='%Y-%m-%d')
    pos_txt = ' · '.join(f"{c[5:].split()[0]} {row[c].iloc[0]:+.3f}" for c in row.columns if c.startswith('pos::'))
    rev = int(row['rev'].iloc[0])
    print(f"✅ live_record 추가: {T.date()} rev {rev}{' (같은 날 재공표 — 이전 행 보존)' if rev else ''} | {pos_txt} "
          f"| 순DV01 {row['net_dv01'].iloc[0]:+,.0f} | 점검 FAIL {row['checks_fail'].iloc[0]} "
          f"| git {row['git'].iloc[0]} | cfg {row['cfg_hash'].iloc[0]} | 누적 {out.index.nunique()}일 {len(out)}행")
    return 0


if __name__ == '__main__':
    sys.exit(main())
