"""
Strategy Factory Dashboard
==========================

3단 구조로 전략 공장 상태를 보여준다:

  1. 탐색 공간  — 생성 가능한 모든 전략 조합 (제너레이터 기준, 스코프 필터 반영)
  2. 이번 달 활성 — 최신(또는 --date) 월 폴더에서 3Y 게이트를 통과해 활성화된 전략
  3. 오늘 진입   — 활성 전략 중 오늘(최신 거래일) 시그널이 켜져 포지션이 잡힌 전략

콘솔: 자산별 요약(후보/활성/진입 수) + 활성 전략 상세 + 전략유형별 분포표
HTML : 모든 조합을 자산별 접이식 표로 전부 나열 (상태 배지 포함)

사용 예:
    python scripts/strategy_dashboard.py                 # 최신 월 폴더
    python scripts/strategy_dashboard.py --date 2026-03-31
    python scripts/strategy_dashboard.py --html          # HTML 대시보드도 저장
    python scripts/strategy_dashboard.py --asset TU1     # 특정 자산만
"""

import argparse
import json
import re
import sys
import html
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:  # 콘솔 한글/기호 깨짐 방지
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.factory.strategy_factory import StrategyFactory
from src.portfolio.selector import StrategySelector, classify_asset_class
from src.strategies.generator import StrategyGenerator


# ─────────────────────────────────────────────────────────────────────────────
# 자산 친화명 (티커 → 사람이 읽기 쉬운 이름)
# ─────────────────────────────────────────────────────────────────────────────
ASSET_NAMES = {
    'TU1 Comdty': '미국 2Y',     'TY1 Comdty': '미국 10Y',
    'DU1 Comdty': '독일 2Y',     'RX1 Comdty': '독일 10Y',
    'G 1 Comdty': '영국 10Y',    'OAT1 Comdty': '프랑스 10Y',
    'IK1 Comdty': '이탈리아 10Y', 'JB1 Comdty': '일본 10Y',
    'YM1 Comdty': '호주 3Y',     'XM1 Comdty': '호주 10Y',
    'KE1 Comdty': '한국 3Y',     'KAA1 Comdty': '한국 10Y',
    'BP1 Curncy': 'GBP',  'AD1 Curncy': 'AUD',  'EC1 Curncy': 'EUR',
    'JY1 Curncy': 'JPY',  'KRW Curncy': 'KRW(원화)',
    'NQ1 Index': '나스닥100',
}

CLASS_ORDER = {'rates': 0, 'fx': 1, 'index': 2, 'other': 3}
CLASS_LABEL = {'rates': '금리', 'fx': '통화(FX)', 'index': '주가지수', 'other': '기타'}

# 상태 코드 (정렬 우선순위 겸용)
ST_ENTER, ST_ACTIVE, ST_STORED, ST_CAND = 0, 1, 2, 3
ST_LABEL = {ST_ENTER: '진입', ST_ACTIVE: '활성', ST_STORED: '저장', ST_CAND: '후보'}


def asset_label(ticker: str) -> str:
    name = ASSET_NAMES.get(ticker)
    return f"{ticker} ({name})" if name else ticker


def short_name(ticker: str) -> str:
    return ASSET_NAMES.get(ticker, ticker)


# ─────────────────────────────────────────────────────────────────────────────
# 전략 한 줄 설명
# ─────────────────────────────────────────────────────────────────────────────
def describe(strategy: dict) -> str:
    """strategy_name + params 로부터 한 줄짜리 한국어 설명을 만든다."""
    name = strategy.get('strategy_name', '')
    p = strategy.get('params', {}) or {}
    rel = strategy.get('related_asset')
    rel_s = short_name(rel) if rel else ''

    g = p.get  # 단축
    try:
        if name == 'ma_crossover':
            return f"{g('fast_period')}/{g('slow_period')}일 단순이평 골든·데드크로스 추세추종"
        if name == 'ema_crossover':
            return f"{g('fast_period')}/{g('slow_period')}일 지수이평 크로스 추세추종"
        if name == 'rsi_momentum':
            return f"RSI({g('period')}) 50선 상향 돌파 모멘텀"
        if name == 'macd_crossover':
            return f"MACD({g('fast_period')}/{g('slow_period')}/{g('signal_period')}) 시그널선 교차"
        if name == 'rate_of_change':
            return f"{g('period')}일 변화율(ROC) 부호 추종"
        if name == 'spread_momentum':
            return f"{rel_s} 대비 스프레드 {g('period')}일 모멘텀"
        if name == 'breakout':
            return f"{g('period')}일 가격 채널(돈치안) 돌파 추종"
        # mean reversion
        if name == 'zscore_reversion':
            return f"{g('period')}일 Z-score ±{g('entry_threshold')} 평균회귀"
        if name == 'rsi_extremes':
            return f"RSI({g('period')}) 과매수({g('overbought')})·과매도({g('oversold')}) 역추세"
        if name == 'bollinger_reversion':
            return f"{g('period')}일 볼린저밴드 {g('std_dev')}σ 이탈 시 역추세"
        if name == 'spread_zscore_reversion':
            return f"{rel_s} 스프레드 Z-score 역추세"
        if name == 'spread_percentile_reversion':
            return f"{rel_s} 스프레드 백분위 역추세"
        # advanced
        if name == 'filtered_momentum':
            return f"{g('ma_period')}일 추세필터 위에서 RSI({g('rsi_period')}) 진입하는 정제 추세추종"
        if name == 'lead_lag_momentum':
            return f"{rel_s}의 {g('period')}일 모멘텀으로 선행 예측 (리드-래그)"
        if name == 'multi_tf_momentum':
            periods = g('periods')
            return f"복수 시간축{periods} 모멘텀이 동의할 때만 진입"
        if name == 'volatility_breakout':
            return f"{g('period')}일 변동성의 {g('k')}배 돌파 시 진입"
        if name == 'relative_strength_rank':
            return f"{g('period')}일 상대강도 상위 {g('top_n')} 종목 추종"
        if name == 'adx_momentum':
            return f"ADX>{g('adx_threshold')} 추세 강할 때만 {g('roc_period')}일 모멘텀 진입"
        if name == 'carry_trade':
            return f"{rel_s} 금리차(캐리) {g('threshold')} 이상이면 고금리 통화 매수"
        # alpha
        if name == 'xsect_momentum':
            return f"{g('universe')} 내 {g('period')}일 수익률 상·하위 {g('top_n')} 롱숏 (크로스섹셔널 모멘텀)"
        if name == 'xsect_carry':
            return f"통화군 금리차 순위 상·하위 {g('top_n')} 롱숏 (크로스섹셔널 캐리)"
        if name == 'curve_to_fx':
            sr = short_name(g('short_rate', '')); lr = short_name(g('long_rate', ''))
            return f"{sr}/{lr} 수익률곡선 기울기로 환율 방향 예측"
        if name == 'rate_diff_to_fx':
            fr = short_name(g('foreign_rate', '')); ur = short_name(g('us_rate', ''))
            return f"{fr}-{ur} 금리차 변화로 환율 방향 예측"
    except Exception:
        pass
    return name or '(설명 없음)'


def params_str(p: dict) -> str:
    """파라미터를 짧은 한 줄로."""
    items = []
    for k, v in sorted((p or {}).items()):
        if isinstance(v, list):
            v = '/'.join(map(str, v))
        elif isinstance(v, str) and v in ASSET_NAMES:
            v = ASSET_NAMES[v]
        items.append(f"{k}={v}")
    return ', '.join(items)


# ─────────────────────────────────────────────────────────────────────────────
# 최신 월 폴더 자동 탐색
# ─────────────────────────────────────────────────────────────────────────────
def resolve_factory_dir(factory_base: Path, date: str = None) -> Path:
    pattern = re.compile(r'^strategies_(\d{4}-\d{2}-\d{2})$')
    dated = []
    for d in factory_base.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m and (d / 'strategies.db').exists():
            dated.append((m.group(1), d))
    if not dated:
        raise FileNotFoundError(f"{factory_base} 안에 strategies.db 를 가진 폴더가 없습니다.")
    dated.sort(key=lambda t: t[0])
    if date:
        for ds, d in dated:
            if ds == date:
                return d
        raise FileNotFoundError(f"strategies_{date} 폴더를 찾을 수 없습니다. "
                                f"가능한 날짜: {', '.join(ds for ds, _ in dated)}")
    return dated[-1][1]


# ─────────────────────────────────────────────────────────────────────────────
# 탐색 공간 + 상태 매칭
# ─────────────────────────────────────────────────────────────────────────────
def combo_key(name, asset, rel, params):
    return (name or '', asset or '', rel or '',
            json.dumps(params or {}, sort_keys=True, default=str))


def build_related_assets_map(loader: DataLoader) -> dict:
    """main.py의 _build_related_assets_map과 동일 (cross_asset_pairs 양방향)."""
    related = {}
    for category, pairs in (loader.get_cross_asset_pairs() or {}).items():
        for pair in pairs:
            if len(pair) != 2:
                continue
            a1, a2 = pair
            related.setdefault(a1, [])
            related.setdefault(a2, [])
            if a2 not in related[a1]:
                related[a1].append(a2)
            if a1 not in related[a2]:
                related[a2].append(a1)
    return related


def build_universe_rows(prices, loader, factory, selector, target_date=None):
    """탐색 공간 전체 조합에 저장/활성/진입 상태를 입힌 행 리스트를 만든다."""
    generator = StrategyGenerator()
    related = build_related_assets_map(loader)
    assets = list(prices.columns)

    # 1) 탐색 공간 전체 (제너레이터가 스코프 필터를 자체 적용)
    rows, by_key = [], {}
    for s in generator.generate_all_strategies(assets, related):
        key = combo_key(s.get('strategy_name'), s.get('asset'),
                        s.get('related_asset'), s.get('params'))
        if key in by_key:        # 혹시 모를 중복 조합은 1개로
            continue
        row = {
            'asset': s['asset'],
            'klass': classify_asset_class(s['asset']),
            'type': s.get('strategy_type', ''),
            'name': s.get('strategy_name', ''),
            'params': s.get('params', {}) or {},
            'desc': describe(s),
            'status': ST_CAND,
            'position': 0.0, 'dir': '-',
            'sharpe_3y': None, 'sharpe_6m': None,
            'date': None,
        }
        rows.append(row)
        by_key[key] = row

    # 2) 이번 달 저장/활성 전략 매칭
    conn = factory._get_conn()
    stored = [factory._row_to_dict(r) for r in
              conn.execute("SELECT * FROM strategies").fetchall()]
    stored = selector._filter_discovery_scope(stored)

    active_rows = []
    for s in stored:
        key = combo_key(s.get('strategy_name'), s.get('asset'),
                        s.get('related_asset'), s.get('params'))
        row = by_key.get(key)
        if row is None:
            # 구설정 잔재 등 탐색 공간 밖의 저장 전략 — 별도 행으로 추가
            row = {
                'asset': s['asset'], 'klass': classify_asset_class(s['asset']),
                'type': s.get('strategy_type', ''), 'name': s.get('strategy_name', ''),
                'params': s.get('params', {}) or {}, 'desc': describe(s) + ' (탐색공간 외)',
                'status': ST_CAND, 'position': 0.0, 'dir': '-',
                'sharpe_3y': None, 'sharpe_6m': None, 'date': None,
            }
            rows.append(row)
            by_key[key] = row
        perf = s.get('performance', {}) or {}
        row['sharpe_3y'] = perf.get('sharpe_3y', s.get('sharpe_3y'))
        row['sharpe_6m'] = perf.get('sharpe_6m', s.get('sharpe_6m'))
        if s.get('is_active'):
            row['status'] = ST_ACTIVE
            active_rows.append((row, s))
        else:
            row['status'] = ST_STORED

    # 3) 활성 전략의 오늘 포지션 → 진입 여부
    for row, s in active_rows:
        try:
            pos_series = selector._compute_position_series(prices, s)
        except Exception:
            pos_series = None
        if pos_series is None or len(pos_series) == 0:
            continue
        if target_date and pd.to_datetime(target_date) in pos_series.index:
            position = float(pos_series.loc[pd.to_datetime(target_date)])
            row['date'] = pd.to_datetime(target_date).date()
        else:
            position = float(pos_series.iloc[-1])
            row['date'] = pos_series.index[-1].date()
        row['position'] = position
        if abs(position) > 1e-9:
            row['status'] = ST_ENTER
            row['dir'] = 'LONG' if position > 0 else 'SHORT'
    return rows


def sort_key(r):
    # 자산군 → 자산 → 상태(진입>활성>저장>후보) → 전략명 → 파라미터
    return (CLASS_ORDER.get(r['klass'], 9), r['asset'], r['status'],
            r['name'], params_str(r['params']))


def group_by_asset(rows):
    by_asset = {}
    for r in rows:
        by_asset.setdefault(r['asset'], []).append(r)
    assets_sorted = sorted(by_asset.keys(),
                           key=lambda a: (CLASS_ORDER.get(classify_asset_class(a), 9), a))
    return by_asset, assets_sorted


def counts(rows):
    c = {ST_ENTER: 0, ST_ACTIVE: 0, ST_STORED: 0, ST_CAND: 0}
    for r in rows:
        c[r['status']] += 1
    return c


def net_direction(grp):
    net = sum(r['position'] * max(r['sharpe_6m'] or 0.0, 0.0)
              for r in grp if r['status'] == ST_ENTER)
    return '▲ 롱' if net > 1e-9 else ('▼ 숏' if net < -1e-9 else '· 중립')


# ─────────────────────────────────────────────────────────────────────────────
# 콘솔 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_console(rows, factory_dir, signal_date):
    rows = sorted(rows, key=sort_key)
    c = counts(rows)
    total = len(rows)

    print("\n" + "=" * 92)
    print(f"  전략 공장 대시보드  |  폴더: {factory_dir.name}  |  기준일: {signal_date}")
    print(f"  탐색 공간 {total:,}개 조합  |  이번 달 활성 {c[ST_ENTER] + c[ST_ACTIVE]}개"
          + (f" (저장만 {c[ST_STORED]}개)" if c[ST_STORED] else "")
          + f"  |  오늘 진입 {c[ST_ENTER]}개")
    print("=" * 92)

    by_asset, assets_sorted = group_by_asset(rows)

    last_klass = None
    for asset in assets_sorted:
        klass = classify_asset_class(asset)
        if klass != last_klass:
            print(f"\n▒▒▒ {CLASS_LABEL.get(klass, klass)} ▒▒▒")
            last_klass = klass
        grp = by_asset[asset]
        gc = counts(grp)
        n_act = gc[ST_ENTER] + gc[ST_ACTIVE]
        line = (f"  ── {asset_label(asset):<24} 후보 {len(grp):>3}개 · "
                f"활성 {n_act:>2}개 · 진입 {gc[ST_ENTER]:>2}개")
        if gc[ST_ENTER]:
            line += f" · 종합 {net_direction(grp)}"
        print(line)
        # 활성(진입 포함)·저장 전략만 상세 표시 — 후보 전체는 HTML에서
        for r in grp:
            if r['status'] == ST_CAND:
                continue
            mark = ST_LABEL[r['status']]
            arrow = {'LONG': '▲L', 'SHORT': '▼S', '-': '  '}[r['dir']]
            sh6 = f"{r['sharpe_6m']:.2f}" if r['sharpe_6m'] is not None else "  - "
            print(f"       [{mark}] {arrow}  {r['type']:<8} {r['name']:<24} "
                  f"Sh6m {sh6:>5}  | {r['desc']}")

    # 전략유형별 분포표
    by_name = {}
    for r in rows:
        d = by_name.setdefault(r['name'], {ST_ENTER: 0, ST_ACTIVE: 0, ST_STORED: 0, ST_CAND: 0,
                                           'type': r['type']})
        d[r['status']] += 1
    print("\n  ── 전략유형별 탐색 공간 ──")
    print(f"     {'전략':<26} {'분류':<9} {'후보':>5} {'활성':>4} {'진입':>4}")
    for name in sorted(by_name, key=lambda n: (by_name[n]['type'], n)):
        d = by_name[name]
        tot = d[ST_ENTER] + d[ST_ACTIVE] + d[ST_STORED] + d[ST_CAND]
        n_act = d[ST_ENTER] + d[ST_ACTIVE]
        print(f"     {name:<26} {d['type']:<9} {tot:>5} {n_act:>4} {d[ST_ENTER]:>4}")
    print("\n" + "=" * 92)


# ─────────────────────────────────────────────────────────────────────────────
# HTML 출력 — 모든 조합 전체 나열 (자산별 접이식)
# ─────────────────────────────────────────────────────────────────────────────
def write_html(rows, factory_dir, signal_date, out_path):
    rows = sorted(rows, key=sort_key)
    c = counts(rows)
    total = len(rows)

    by_asset, assets_sorted = group_by_asset(rows)

    css = """
    body{font-family:'Malgun Gothic',system-ui,sans-serif;background:#0f1116;color:#e6e6e6;margin:0;padding:24px;}
    h1{font-size:20px;margin:0 0 4px;} .meta{color:#9aa0aa;font-size:13px;margin-bottom:18px;}
    .klass{margin:26px 0 8px;font-size:15px;color:#8ab4f8;border-bottom:1px solid #2a2f3a;padding-bottom:4px;}
    details{margin:8px 0;} summary{cursor:pointer;font-weight:600;font-size:14px;padding:6px 4px;}
    summary:hover{background:#171b24;} summary .net{font-weight:400;color:#9aa0aa;margin-left:8px;}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:4px 0 10px;}
    td,th{padding:4px 8px;text-align:left;border-bottom:1px solid #1d2230;}
    th{color:#9aa0aa;font-weight:500;position:sticky;top:0;background:#0f1116;}
    .pill{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;font-weight:700;}
    .st0{background:#10381f;color:#34d399;} .st1{background:#16243f;color:#8ab4f8;}
    .st2{background:#3a2e12;color:#fbbf24;} .st3{background:#1c1f27;color:#5d6470;}
    tr.cand td{color:#5d6470;} tr.cand .pill{font-weight:400;}
    .long{color:#34d399;} .short{color:#f87171;} .flat{color:#555c6b;}
    .type{color:#c4b5fd;} .sh{color:#9aa0aa;font-variant-numeric:tabular-nums;}
    .desc{color:#cbd2da;} tr.cand .desc, tr.cand .type{color:#5d6470;}
    .params{color:#7a828e;font-size:11.5px;}
    """
    st_pill = {ST_ENTER: ("st0", "진입"), ST_ACTIVE: ("st1", "활성"),
               ST_STORED: ("st2", "저장"), ST_CAND: ("st3", "후보")}

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>"]
    parts.append("<h1>전략 공장 대시보드</h1>")
    parts.append(f"<div class='meta'>폴더 {html.escape(factory_dir.name)} · 기준일 {signal_date} · "
                 f"탐색 공간 <b>{total:,}개</b> 조합 · 이번 달 활성 "
                 f"<b class='st1'>{c[ST_ENTER] + c[ST_ACTIVE]}개</b>"
                 + (f" · 저장만 {c[ST_STORED]}개" if c[ST_STORED] else "")
                 + f" · 오늘 진입 <b class='st0'>{c[ST_ENTER]}개</b></div>")

    last_klass = None
    for asset in assets_sorted:
        klass = classify_asset_class(asset)
        if klass != last_klass:
            parts.append(f"<div class='klass'>{CLASS_LABEL.get(klass, klass)}</div>")
            last_klass = klass
        grp = by_asset[asset]
        gc = counts(grp)
        n_act = gc[ST_ENTER] + gc[ST_ACTIVE]
        open_attr = " open" if n_act else ""
        net_html = f" · 종합 {net_direction(grp)}" if gc[ST_ENTER] else ""
        parts.append(f"<details{open_attr}><summary>{html.escape(asset_label(asset))}"
                     f"<span class='net'>후보 {len(grp)}개 · 활성 {n_act}개 · "
                     f"진입 {gc[ST_ENTER]}개{net_html}</span></summary>")
        parts.append("<table><tr><th>상태</th><th>방향</th><th>분류</th><th>전략</th>"
                     "<th>파라미터</th><th>Sh3Y</th><th>Sh6M</th><th>설명</th></tr>")
        for r in grp:
            cls, txt = st_pill[r['status']]
            tr_cls = " class='cand'" if r['status'] == ST_CAND else ""
            dcls = {'LONG': 'long', 'SHORT': 'short', '-': 'flat'}[r['dir']]
            dtxt = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '·'}[r['dir']]
            sh3 = f"{r['sharpe_3y']:.2f}" if r['sharpe_3y'] is not None else "–"
            sh6 = f"{r['sharpe_6m']:.2f}" if r['sharpe_6m'] is not None else "–"
            parts.append(
                f"<tr{tr_cls}><td><span class='pill {cls}'>{txt}</span></td>"
                f"<td class='{dcls}'>{dtxt}</td>"
                f"<td class='type'>{html.escape(r['type'])}</td>"
                f"<td>{html.escape(r['name'])}</td>"
                f"<td class='params'>{html.escape(params_str(r['params']))}</td>"
                f"<td class='sh'>{sh3}</td><td class='sh'>{sh6}</td>"
                f"<td class='desc'>{html.escape(r['desc'])}</td></tr>")
        parts.append("</table></details>")
    parts.append("</body></html>")
    out_path.write_text("".join(parts), encoding='utf-8')
    print(f"\n📊 HTML 대시보드 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="전략 공장 대시보드")
    ap.add_argument('--date', default=None, help="월 폴더 날짜 (예: 2026-03-31). 미지정 시 최신")
    ap.add_argument('--asset', default=None, help="특정 자산만 필터 (티커 일부, 예: TU1)")
    ap.add_argument('--signal-date', default=None, help="시그널 기준일 (미지정 시 최신 거래일)")
    ap.add_argument('--html', action='store_true', help="HTML 대시보드도 저장")
    args = ap.parse_args()

    factory_base = Path(__file__).parent.parent / 'src' / 'factory'
    factory_dir = resolve_factory_dir(factory_base, args.date)

    # 데이터 로드
    print("📊 가격 데이터 로딩...")
    loader = DataLoader()
    prices_raw = loader.load_data(use_cache=True)
    prices = DataPreprocessor(prices_raw).clean().get_data()

    factory = StrategyFactory(storage_dir=str(factory_dir))
    selector = StrategySelector(factory=factory)

    rows = build_universe_rows(prices, loader, factory, selector,
                               target_date=args.signal_date)
    if args.asset:
        rows = [r for r in rows if args.asset.lower() in r['asset'].lower()]
    if not rows:
        print("⚠️ 표시할 전략 조합이 없습니다.")
        factory.close()
        return

    signal_date = max((r['date'] for r in rows if r['date']), default='N/A')

    print_console(rows, factory_dir, signal_date)

    if args.html:
        out = Path(f"strategy_dashboard_{factory_dir.name.replace('strategies_', '')}.html")
        write_html(rows, factory_dir, signal_date, out)

    factory.close()


if __name__ == '__main__':
    main()
