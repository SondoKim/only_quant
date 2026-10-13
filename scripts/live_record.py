# -*- coding: utf-8 -*-
"""라이브 기록 — 그날 '공표한' 금리 북 목표를 data/live_record.csv 에 추가 (2026-10-13).

왜 필요한가: sleeve_backtest_log.csv 는 매 실행마다 전체 이력을 현재 모델·현재 데이터로 다시
쓴다 (2026-10-13 감사로 과거 SR 이 1.24 → 0.72 로 바뀐 것처럼). 그래서 '그날 실제로 무엇을
권고했는가'는 어디에도 남지 않았다. 이 파일은 날짜별로 한 행씩 쌓고 과거 행을 고치지 않는다 —
실매매 대사(scripts/reconcile_live.py)와 사후 검증의 기준이다.

행 = 데이터 기준일 T (같은 T 재실행이면 그 행만 최신 실행으로 교체, 과거 행 불변):
  generated · git(커밋, dirty 표시) · cfg_hash(운용 설정 지문) · data_asof
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
    row = {'generated': pd.Timestamp.now().isoformat(timespec='seconds'), 'git': _git(),
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
    st = e.book_stop_status() or {}
    row.update({'net_dv01': round(net, 1), 'gross_dv01': round(gross, 1),
                'stop_scale': st.get('scale'), 'stop_dd': st.get('dd'),
                'dv01_factor': (float(e._dv01_state['factor'].iloc[-1]) if e._dv01_state else None)})
    return pd.DataFrame([row], index=pd.Index([T.date()], name='date'))


def main() -> int:
    row = build_row()
    T = row.index[0]
    if RECORD.exists():
        old = pd.read_csv(RECORD, index_col='date', parse_dates=False)
        old.index = pd.to_datetime(old.index).date
        later = [d for d in old.index if d > T]
        if later:
            print(f"⚠ live_record 에 {T} 이후 행이 있음 ({later[-1]}) — 데이터가 뒤로 간 실행, 기록하지 않음")
            return 0
        replaced = T in old.index
        out = pd.concat([old[old.index != T], row])
    else:
        replaced, out = False, row
    out.index.name = 'date'
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    out.sort_index().to_csv(RECORD)
    pos_txt = ' · '.join(f"{c[5:].split()[0]} {row[c].iloc[0]:+.3f}" for c in row.columns if c.startswith('pos::'))
    print(f"✅ live_record {'갱신' if replaced else '추가'}: {T} | {pos_txt} | 순DV01 {row['net_dv01'].iloc[0]:+,.0f} "
          f"| git {row['git'].iloc[0]} | cfg {row['cfg_hash'].iloc[0]} | 누적 {len(out)}일")
    return 0


if __name__ == '__main__':
    sys.exit(main())
