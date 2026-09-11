"""금리 팩터 매매 후보 요약 → data/factor_summary.json (2026-09-11).

대시보드(total_dashboard '금리 팩터', portfolio_management)가 각자 계산하던 후처리를 생산자 쪽으로
옮겼다 — 소비자는 이 JSON 을 읽어 표시만 한다. 시그널 시계열 차트는 각 대시보드가 테마(화이트/블랙)에
맞춰 따로 그리므로 sleeve_factor_signals.csv 는 그대로 원천으로 남는다.

입력   sleeve_factor_signals.csv (run_sleeve_backtest.py 산출, 컬럼 = 팩터::티커)
       data/cache/prices_*.parquet 최신 1개 (최근 20일 성과·스파크라인용 선물 종가)
       config/factor_meta.json (표시용 상수: top_n, 창, 라벨, 정의문)
출력   data/factor_summary.json
  {generated, asof, sources{signals, prices}, meta{...factor_meta 그대로...},
   factors{key: {last{티커: z}, longs[...], shorts[...], pairs[...]}}}
  longs/shorts 항목  {label, z, ticker, perf20, series[진입 6일 전~현재 오리엔티드 누적수익률]}
  pairs 항목         {long_label, long_z, long_ticker, short_label, short_z, short_ticker, spread, perf20, series}
  방향성(trend·policy)은 longs/shorts, 상대매매(value·carry)는 pairs 만 채운다. policy 는 국가별 1개(테너 라벨).

실행   python scripts/factor_summary.py   (daily_run.py 가 1단계 뒤 두 모드 모두에서 호출)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIGNALS_CSV = ROOT / 'sleeve_factor_signals.csv'
CACHE_DIR = ROOT / 'data' / 'cache'
META_PATH = ROOT / 'config' / 'factor_meta.json'
OUT_PATH = ROOT / 'data' / 'factor_summary.json'


def load_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding='utf-8'))


def load_signals(start: str | None = None) -> dict[str, pd.DataFrame]:
    """{팩터키: DataFrame(index=date, columns=티커)} — curve 는 대시보드 미표시(weight 0)라 제외."""
    if not SIGNALS_CSV.exists():
        return {}
    df = pd.read_csv(SIGNALS_CSV, parse_dates=['date']).set_index('date').sort_index()
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    out = {}
    for key in ('trend', 'value', 'carry', 'policy'):
        cols = [c for c in df.columns if c.startswith(f'{key}::')]
        if not cols:
            continue
        sub = df[cols].rename(columns={c: c.split('::', 1)[1] for c in cols}).dropna(how='all')
        if not sub.empty:
            out[key] = sub
    return out


def latest_prices_path() -> Path | None:
    pcs = sorted(CACHE_DIR.glob('prices_*.parquet'), key=lambda p: p.stat().st_mtime)
    return pcs[-1] if pcs else None


def _last_row(sig_df: pd.DataFrame):
    valid = sig_df.dropna(how='all')
    return valid.iloc[-1] if len(valid) else None


def _oriented_single(prices, ticker, z, window, ctx):
    """방향성 후보 — 진입시점(window 일 전) 리베이스, z 부호 반영(위로 갈수록 이기는 방향)."""
    if prices is None or ticker not in prices.columns:
        return None
    s = prices[ticker].dropna()
    if len(s) < window + 1:
        return None
    tail = s.tail(window + 1 + ctx)
    if len(tail) < window + 1:
        tail = s.tail(window + 1)
    ret = tail / tail.iloc[-(window + 1)] - 1.0
    return -ret if z < 0 else ret


def _oriented_pair(prices, tl, ts, window, ctx):
    """상대매매 후보 — 롱레그 수익률 − 숏레그 수익률, 진입시점 리베이스."""
    if prices is None or tl not in prices.columns or ts not in prices.columns:
        return None
    sl, ss = prices[tl].dropna(), prices[ts].dropna()
    idx = sl.index.intersection(ss.index).sort_values()
    if len(idx) < window + 1:
        return None
    tail_idx = idx[-(window + 1 + ctx):]
    if len(tail_idx) < window + 1:
        tail_idx = idx[-(window + 1):]
    sl, ss = sl.loc[tail_idx], ss.loc[tail_idx]
    return (sl / sl.iloc[-(window + 1)] - 1.0) - (ss / ss.iloc[-(window + 1)] - 1.0)


def _perf(series):
    if series is None or len(series) < 2:
        return None, None
    return round(float(series.iloc[-1]), 6), [round(float(v), 6) for v in series.values]


def top_n(sig_df, names, n):
    """(longs, shorts) — 각 [(라벨, z, 티커)]. longs = z>0 상위, shorts = z<0 하위."""
    row = _last_row(sig_df)
    if row is None:
        return [], []
    vals = [(names.get(a, a), float(v), a) for a, v in row.items() if pd.notna(v)]
    longs = sorted((t for t in vals if t[1] > 0), key=lambda x: -x[1])[:n]
    shorts = sorted((t for t in vals if t[1] < 0), key=lambda x: x[1])[:n]
    return longs, shorts


def top_pairs(sig_df, names, n):
    """[(롱명, z롱, 숏명, z숏, 스프레드, 롱티커, 숏티커)] — z 최상위↔최하위 짝, 격차 큰 순."""
    row = _last_row(sig_df)
    if row is None:
        return []
    vals = sorted(((names.get(a, a), float(v), a) for a, v in row.items() if pd.notna(v)), key=lambda x: -x[1])
    pairs = []
    for i in range(min(n, len(vals) // 2)):
        ln, zl, tl = vals[i]
        sn, zs, ts = vals[-1 - i]
        sp = zl - zs
        if sp <= 0:
            break
        pairs.append((ln, zl, sn, zs, sp, tl, ts))
    return pairs


def top_policy(sig_df, policy_group, n):
    """Policy — 국가별 중복(동일 정책금리) 제거 + 테너 라벨. 반환은 top_n 과 같은 꼴."""
    row = _last_row(sig_df)
    if row is None:
        return [], []
    seen = {}
    for a, v in row.items():
        if pd.isna(v):
            continue
        grp = policy_group.get(a)
        if grp is None or grp[0] in seen:
            continue
        seen[grp[0]] = (f'{grp[0]} ({grp[1]})', float(v), a)
    vals = list(seen.values())
    longs = sorted((t for t in vals if t[1] > 0), key=lambda x: -x[1])[:n]
    shorts = sorted((t for t in vals if t[1] < 0), key=lambda x: x[1])[:n]
    return longs, shorts


def build(meta: dict | None = None) -> dict:
    meta = meta or load_meta()
    n, win, ctx = int(meta['top_n']), int(meta['perf_window']), int(meta['perf_ctx'])
    names, pgroup = meta['asset_names'], meta['policy_group']
    signals = load_signals(meta.get('perf_start'))
    ppath = latest_prices_path()
    prices = pd.read_parquet(ppath) if ppath else None
    factors, asof = {}, None
    for key in meta['factor_keys']:
        sig = signals.get(key)
        if sig is None:
            continue
        valid = sig.dropna(how='all')
        d = valid.index[-1]
        asof = max(asof, d) if asof is not None else d
        entry = {'asof': d.strftime('%Y-%m-%d'),
                 'last': {a: round(float(v), 6) for a, v in valid.iloc[-1].items() if pd.notna(v)},
                 'longs': [], 'shorts': [], 'pairs': []}
        if key in meta['pair_factors']:
            for ln, zl, sn, zs, sp, tl, ts in top_pairs(sig, names, n):
                perf, series = _perf(_oriented_pair(prices, tl, ts, win, ctx))
                entry['pairs'].append({'long_label': ln, 'long_z': round(zl, 6), 'long_ticker': tl,
                                       'short_label': sn, 'short_z': round(zs, 6), 'short_ticker': ts,
                                       'spread': round(sp, 6), 'perf20': perf, 'series': series})
        else:
            longs, shorts = top_policy(sig, pgroup, n) if key == 'policy' else top_n(sig, names, n)
            for side, items in (('longs', longs), ('shorts', shorts)):
                for label, z, t in items:
                    perf, series = _perf(_oriented_single(prices, t, z, win, ctx))
                    entry[side].append({'label': label, 'z': round(z, 6), 'ticker': t, 'perf20': perf, 'series': series})
        factors[key] = entry
    return {'generated': datetime.now().isoformat(timespec='seconds'),
            'asof': asof.strftime('%Y-%m-%d') if asof is not None else None,
            'sources': {'signals': str(SIGNALS_CSV.relative_to(ROOT)),
                        'prices': str(ppath.relative_to(ROOT)) if ppath else None},
            'meta': meta, 'factors': factors}


def main() -> int:
    if not SIGNALS_CSV.exists():
        print(f'[factor_summary] {SIGNALS_CSV.name} 없음 — 건너뜀')
        return 1
    out = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    counts = {k: (len(v['pairs']) or f"{len(v['longs'])}L/{len(v['shorts'])}S") for k, v in out['factors'].items()}
    print(f"[factor_summary] asof {out['asof']}  {counts}  → {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
