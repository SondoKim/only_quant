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
        # alpha4 (yield-based)
        if name == 'rates_carry':
            return f"커브 기울기(10Y-2Y) {g('threshold')}%p 이상이면 듀레이션 롱 (캐리+롤다운)"
        if name == 'rates_value':
            return f"자국 금리의 {g('lookback')}일 Z-score ±{g('entry_z')} 이탈 시 역추세 (채권 밸류)"
        if name == 'real_rate_fx':
            return f"실제 2Y 금리차의 {g('period')}일 변화로 환율 방향 예측 (실금리 캐리)"
        if name == 'policy_momentum':
            return f"자국 2Y 금리의 {g('period')}일 변화로 정책 사이클 추종 (인하시 듀레이션 롱)"
        if name == 'month_end_seasonal':
            return f"월말 {g('days_before')}영업일 전부터 듀레이션 연장 매수 (월말 시즈널)"
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
            'related': s.get('related_asset'),
            'desc': describe(s),
            'status': ST_CAND,
            'position': 0.0, 'dir': '-',
            'sharpe_3y': None, 'sharpe_6m': None,
            'date': None, 'picked': False,
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
                'params': s.get('params', {}) or {},
                'related': s.get('related_asset'),
                'desc': describe(s) + ' (탐색공간 외)',
                'status': ST_CAND, 'position': 0.0, 'dir': '-',
                'sharpe_3y': None, 'sharpe_6m': None, 'date': None, 'picked': False,
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


def group_by_strat(rows):
    by_strat = {}
    for r in rows:
        by_strat.setdefault(r['name'], []).append(r)
    TYPE_ORDER = {'momentum': 0, 'mean_reversion': 1, 'advanced': 2, 'alpha1': 3, 'alpha2': 4, 'alpha4': 5}
    strats_sorted = sorted(by_strat.keys(),
                           key=lambda n: (TYPE_ORDER.get(by_strat[n][0]['type'], 9), n))
    return by_strat, strats_sorted


STATIC_STRATEGIES = [
    {"name": "리버전", "asset": "3선, 10선", "desc": "전일 미국장 금리방향과 반대로 진입", "overnight": "X", "status": "ON"},
    {"name": "채널", "asset": "KRW, JPY", "desc": "FX 가격 밴드 스윙 트레이딩", "overnight": "O", "status": "ON"},
    {"name": "PCA금리", "asset": "미국, 독일금리", "desc": "이론금리 괴리분에 대한 회귀 가정 매매", "overnight": "O", "status": "모의"},
    {"name": "아이언콘돌", "asset": "미국금리옵션", "desc": "금리 옵션 양매도+양매수", "overnight": "O", "status": "모의"},
    {"name": "변동성 돌파 전략", "asset": "미국금리, JPY, G", "desc": "전일 변동성 range 당일 돌파 시 매매", "overnight": "X", "status": "OFF"},
    {"name": "FX캐리극대화 전략", "asset": "글로벌 FX", "desc": "캐리 극대화 fx long-short", "overnight": "O", "status": "OFF"}
]


QUANT_STRATS_INFO = {
    # 고급/다자산 전략 (Advanced)
    'filtered_momentum': {
        'name': '추세 RSI 전략 (Advanced)',
        'asset': '글로벌 금리, FX',
        'desc': 'm일 추세필터 위에서 RSI(n) 진입하는 정제 추세추종',
        'overnight': 'O'
    },
    'lead_lag_momentum': {
        'name': '선행 지수 모멘텀 전략 (Advanced)',
        'asset': '글로벌 금리, FX',
        'desc': '선행 자산(예: 나스닥 등)의 p일 모멘텀을 활용한 후행 자산 방향 예측',
        'overnight': 'O'
    },
    'multi_tf_momentum': {
        'name': '다중 시간축 추세 전략 (Advanced)',
        'asset': '글로벌 금리, FX',
        'desc': '단기/중기/장기 다중 시간축 모멘텀이 일치할 때 추세 진입',
        'overnight': 'O'
    },
    'relative_strength_rank': {
        'name': '상대강도 로테이션 전략 (Advanced)',
        'asset': '글로벌 금리, FX',
        'desc': 'p일 상대강도(RS) 순위 기반 상위 자산 편입 및 모멘텀 추종',
        'overnight': 'O'
    },
    'volatility_breakout': {
        'name': '변동성 돌파 전략 (Advanced)',
        'asset': '미국금리, JPY, G',
        'desc': '전일 가격 변동성 range의 k배 당일 돌파 시 추세 추종',
        'overnight': 'X'
    },
    # 알파1 전략
    'xsect_momentum': {
        'name': '크로스섹셔널 모멘텀 전략 (Alpha1)',
        'asset': '글로벌 금리, FX',
        'desc': '자산군 내 상대 수익률 순위 기반 롱숏 포트폴리오',
        'overnight': 'O'
    },
    'xsect_carry': {
        'name': '크로스섹셔널 캐리 전략 (Alpha1)',
        'asset': '글로벌 FX',
        'desc': '통화군 금리차(캐리) 순위 기반 고금리 롱 + 저금리 숏',
        'overnight': 'O'
    },
    # 알파2 전략
    'curve_to_fx': {
        'name': '커브 기울기 환율 예측 전략 (Alpha2)',
        'asset': '글로벌 금리, FX',
        'desc': '국채 금리 커브(10Y-2Y) 기울기 변화로 관련 통화 방향 예측',
        'overnight': 'O'
    },
    'rate_diff_to_fx': {
        'name': '금리차 환율 예측 전략 (Alpha2)',
        'asset': '글로벌 금리, FX',
        'desc': '국가 간 금리차 변화로 양국 통화 환율 방향 예측',
        'overnight': 'O'
    },
    # 알파4 전략
    'rates_carry': {
        'name': '채권 롤다운 전략 (Alpha4)',
        'asset': '글로벌 금리',
        'desc': '금리 커브 기울기 임계치 초과 시 장기채 매수 및 롤다운 효과 획득',
        'overnight': 'O'
    },
    'rates_value': {
        'name': '채권 밸류에이션 전략 (Alpha4)',
        'asset': '글로벌 금리',
        'desc': '자국 금리의 역사적 평균 대비 괴리(Z-score) 해소 시 평균회귀 매매',
        'overnight': 'O'
    },
    'real_rate_fx': {
        'name': '실질금리 환율 예측 전략 (Alpha4)',
        'asset': '글로벌 금리, FX',
        'desc': '국가 간 실질 금리차(2Y) 변화로 환율 방향 예측',
        'overnight': 'O'
    },
    'policy_momentum': {
        'name': '통화정책 모멘텀 전략 (Alpha4)',
        'asset': '글로벌 금리',
        'desc': '단기 금리(2Y) 변화율 추종으로 정책 금리 사이클 추종 매매',
        'overnight': 'O'
    },
    'month_end_seasonal': {
        'name': '채권 월말 효과 전략 (Alpha4)',
        'asset': '글로벌 금리',
        'desc': '월말 지수 리밸런싱에 따른 듀레이션 확대 효과 추종 선매수',
        'overnight': 'O'
    }
}





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
# 슬리브 엔진 (금리 북) — 슬리브별 → 자산별 현황
# ─────────────────────────────────────────────────────────────────────────────
SLEEVE_INFO = {
    'trend':  ('추세 (TSMOM)',     '6/12개월 가격 추세 z-score · 레짐게이트로 횡보장 차단 · 방향성'),
    'value':  ('밸류 (평균회귀)',  '2년 평균 대비 가격 괴리 — 과열 숏 / 과매도 롱 · 시장중립'),
    'carry':  ('캐리 (일드 레벨)', '국가간 장기 일드 레벨 z — 고금리국 롱 / 저금리국 숏 · 시장중립'),
    'curve':  ('커브 캐리',        '10Y−2Y 기울기 z — 스팁 롱 / 역전 숏 · 방향성'),
    'policy': ('정책 모멘텀',      '2Y 금리 6개월 변화 — 인하 사이클 롱 / 인상 사이클 숏 · 방향성'),
    'reversion': ('단기 평균회귀',  '10일 무브의 글로벌 평균 대비 과열/과매도 페이드 · 시장중립 · 상시가동(스탑 미적용)'),
}
SLEEVE_ORDER = ['trend', 'value', 'carry', 'curve', 'policy', 'reversion']

# 한눈 표용 슬리브 전략명 (사용자 명명 규칙: 팩터별 전략 이름 + 설명)
SLEEVE_STRATS = [
    ('trend',  '글로벌 금리 추세 전략 (Sleeve)',
     '6/12개월 가격 추세 z-score 추종 · Hurst 레짐게이트로 횡보장 자동 차단'),
    ('value',  '글로벌 금리 밸류 전략 (Sleeve)',
     '2년 평균 대비 가격 괴리 평균회귀 — 과열 숏 / 과매도 롱 (시장중립)'),
    ('carry',  '글로벌 금리 캐리 전략 (Sleeve)',
     '국가간 장기 일드 레벨 비교 — 고금리국 롱 / 저금리국 숏 (시장중립)'),
    ('curve',  '커브 캐리 전략 (Sleeve)',
     '자국 10Y−2Y 기울기 z — 스팁 롱 / 역전 숏 (A/B 결과 미채택)'),
    ('policy', '통화정책 사이클 전략 (Sleeve)',
     '2Y 금리 6개월 변화로 중앙은행 인하/인상 사이클 추종'),
    ('reversion', '금리 단기 평균회귀 전략 (Sleeve)',
     '10일 무브의 글로벌 듀레이션 평균 대비 괴리 페이드 — 횡보장 수익원, 시장중립, 패시브 집행 필수'),
]


def build_sleeve_snapshot(loader):
    """SleeveEngine 금리 북 스냅샷: 슬리브별 최신 시그널 + 최종 목표 포지션.

    main._merge_sleeve_rates 와 동일 구성(2010+ 패널, position_smooth 적용).
    실패 시 None — 대시보드는 공장 뷰만으로 동작.
    """
    try:
        import yaml as _yaml
        from src.sleeves.sleeve_engine import SleeveEngine
        cfg_path = Path(__file__).parent.parent / 'config' / 'indicators.yaml'
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = (_yaml.safe_load(f) or {}).get('sleeves', {}) or {}
        px = DataPreprocessor(
            loader.load_data(start_date='2010-01-01', use_cache=True)
        ).clean().get_data()
        yields = loader.load_signal_yields(start_date='2010-01-01', use_cache=True)
        engine = SleeveEngine(px, config=cfg, yields=yields)
        return engine.sleeve_snapshot('rates')
    except Exception as e:
        print(f"⚠️ 슬리브 스냅샷 생성 실패 (금리 북 섹션 생략): {e}")
        return None


def _sleeve_dir(v: float, thresh: float = 0.02) -> str:
    return 'LONG' if v > thresh else ('SHORT' if v < -thresh else '-')


def sleeve_signal_rows(snap, delta_per_unit=None):
    """스냅샷 → 공식 시그널 테이블용 금리 행 (main.py SLEEVE 행과 동일).

    delta_per_unit: 포지션 1.0당 만원 환산 계수 (--delta-budget 캘리브레이션).
    """
    rows = []
    for a in sorted(snap['target'], key=lambda x: (CLASS_ORDER.get(classify_asset_class(x), 9), x)):
        p = snap['target'][a]
        q = snap.get('prev', {}).get(a, p)
        r = {'asset': a, 'klass': classify_asset_class(a),
             'dir': _sleeve_dir(p), 'conf': abs(p), 'pos': p,
             'n': 0, 'src': 'sleeve', 'prev': q, 'dpos': p - q}
        if delta_per_unit:
            r['delta_w'] = p * delta_per_unit
            r['ddelta_w'] = (p - q) * delta_per_unit
        rows.append(r)
    return rows


# FX 선물 가격 → 통상 호가 컨벤션 환산
FX_QUOTE = {
    'JY1 Curncy': ('USDJPY', lambda p: 10000.0 / p),
    'BP1 Curncy': ('GBPUSD', lambda p: p / 100.0),
    'AD1 Curncy': ('AUDUSD', lambda p: p / 100.0),
    'EC1 Curncy': ('EURUSD', lambda p: p),
    'KRW Curncy': ('USDKRW', lambda p: p),
}


def attach_underlying(sig_rows, prices):
    """각 행에 기초지표 부착: 금리=해당 테너 국채 일드(%), FX=통상 호가."""
    import yaml as _yaml
    try:
        assets_path = Path(__file__).parent.parent / 'config' / 'assets.yaml'
        with open(assets_path, 'r', encoding='utf-8') as f:
            ymap = ((_yaml.safe_load(f) or {}).get('signal_yields', {})
                    or {}).get('tradeable_yield_map', {}) or {}
    except Exception:
        ymap = {}
    last = prices.iloc[-1]
    for r in sig_rows:
        a = r['asset']
        if r['klass'] == 'rates':
            yt = ymap.get(a)
            if yt and yt in last.index and last[yt] == last[yt]:
                r['underlying'] = f"{float(last[yt]):.3f}%"
        elif r['klass'] == 'fx' and a in FX_QUOTE and a in last.index:
            name, fn = FX_QUOTE[a]
            try:
                r['underlying'] = f"{name} {_px_fmt(fn(float(last[a])))}"
            except ZeroDivisionError:
                pass


def _px_fmt(v):
    """자산 가격 표시용 자릿수 자동 포맷."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '-'
    if v != v:
        return '-'
    if abs(v) >= 1000:
        return f"{v:,.1f}"
    if abs(v) >= 100:
        return f"{v:.2f}"
    if abs(v) >= 10:
        return f"{v:.3f}"
    return f"{v:.4f}"


def compute_delta_info(snap, budget_w, gross_budget_w=None):
    """순델타/그로스 한도(만원) → 환산 계수 C 역산.

    C = min(순한도/과거최대|순포지션|, 그로스한도/과거최대 그로스) — 역사상
    최악의 날에도 두 한도를 모두 만족하는 가장 큰 환산 계수.
    """
    net_hist = snap.get('net_hist')
    if net_hist is None or len(net_hist) == 0:
        return None
    max_net = float(net_hist.abs().max())
    if max_net <= 0:
        return None
    c = budget_w / max_net
    gross_hist = snap.get('gross_hist')
    max_gross = float(gross_hist.max()) if gross_hist is not None and len(gross_hist) else 0.0
    binding = 'net'
    if gross_budget_w and max_gross > 0:
        c_gross = gross_budget_w / max_gross
        if c_gross < c:
            c, binding = c_gross, 'gross'
    net_w = sum(snap['target'].values()) * c
    gross_w = sum(abs(v) for v in snap['target'].values()) * c
    return {'per_unit': c, 'budget': budget_w, 'gross_budget': gross_budget_w,
            'net_w': net_w, 'gross_w': gross_w,
            'usage': abs(net_w) / budget_w,
            'gross_usage': (gross_w / gross_budget_w) if gross_budget_w else None,
            'max_net_units': max_net, 'max_gross_units': max_gross,
            'binding': binding}


def print_sleeve_console(snap):
    d = snap['date'].date() if hasattr(snap['date'], 'date') else snap['date']
    print(f"\n▌ 슬리브 엔진 — 금리 북  (기준일 {d} · 연속 시그널 · 볼타게팅 — "
          f"main.py --mode signals 금리 SLEEVE 행과 동일)")
    assets = sorted(snap['target'], key=lambda x: x)
    for name in SLEEVE_ORDER:
        if name not in snap['sleeves']:
            continue
        s = snap['sleeves'][name]
        title, desc = SLEEVE_INFO.get(name, (name, ''))
        print(f"  ── {title:<14} (가중 {s['weight']:.1f})  {desc}")
        line = []
        for a in assets:
            v = s['signals'].get(a)
            if v is None:
                continue
            arrow = {'LONG': '▲', 'SHORT': '▼', '-': '·'}[_sleeve_dir(v, 0.1)]
            line.append(f"{short_name(a)} {arrow}{v:+.2f}")
        for i in range(0, len(line), 6):
            print("       " + "  ".join(f"{x:<14}" for x in line[i:i + 6]))
    print(f"  ── {'최종 목표':<14} (인버스볼 × 포트 볼타게팅 × 스무딩 — 주문 기준)")
    line = []
    for a in assets:
        p = snap['target'][a]
        arrow = {'LONG': '▲', 'SHORT': '▼', '-': '·'}[_sleeve_dir(p)]
        line.append(f"{short_name(a)} {arrow}{p:+.2f}")
    for i in range(0, len(line), 6):
        print("       " + "  ".join(f"{x:<14}" for x in line[i:i + 6]))


def load_fx_pos_history(n=120):
    """trading_log.csv(공장 워크포워드 백테스트)의 FX 일별 포지션 최근 n일.

    공식 피드와 같은 파이프라인의 기록이라 진입/청산 패턴 확인용으로 정합 —
    단, 마지막 백테스트 실행 시점까지만 반영. 로그 없으면 None.
    """
    try:
        root = Path(__file__).parent.parent
        df = pd.read_csv(root / 'trading_log.csv', parse_dates=['Date']).set_index('Date')
        cols = {c[:-4]: c for c in df.columns if c.endswith('_Pos') and 'Curncy' in c}
        if not cols:
            return None
        out = df[list(cols.values())].tail(n)
        out.columns = list(cols.keys())
        return out
    except Exception:
        return None


def build_ytd_perf_html(start='2026-01-01'):
    """YTD 성과 plotly div: trading_log.csv(FX 공장) + sleeve_backtest_log.csv(금리
    슬리브)를 combined_report.py 와 동일하게 10% 볼 정규화 후 누적 PnL 곡선.
    로그가 없거나 plotly 미설치면 None (대시보드는 차트 없이 동작)."""
    try:
        import numpy as np
        import plotly.graph_objects as go
        root = Path(__file__).parent.parent
        fac = pd.read_csv(root / 'trading_log.csv', parse_dates=['Date']).set_index('Date')
        slv = pd.read_csv(root / 'sleeve_backtest_log.csv', index_col=0, parse_dates=True)
        df = pd.concat([fac['total_fx_cumpnl'].diff().rename('FX'),
                        (slv['rates'] * 100.0).rename('RATES')],
                       axis=1, join='inner').fillna(0.0)
        tv = 0.10 / np.sqrt(252) * 100.0
        for col in ['FX', 'RATES']:
            sd = df[col].std()
            if sd > 0:
                df[col] = df[col] * (tv / sd)
        df['COMBINED'] = df[['FX', 'RATES']].mean(axis=1)
        ytd = df[df.index >= pd.to_datetime(start)]
        if len(ytd) < 5:
            return None
        colors = {'FX': '#f0a23c', 'RATES': '#4c8ef7', 'COMBINED': '#34d399'}
        fig = go.Figure()
        for col in ['FX', 'RATES', 'COMBINED']:
            s = ytd[col]
            sr = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0
            cum = s.cumsum().round(3)
            fig.add_trace(go.Scatter(
                x=ytd.index, y=cum, name=f"{col} (SR {sr:.2f})",
                line=dict(width=3.0 if col == 'COMBINED' else 1.6, color=colors[col])))
            # 최신 날짜 값 라벨
            fig.add_annotation(
                x=ytd.index[-1], y=float(cum.iloc[-1]),
                text=f"<b>{cum.iloc[-1]:+.2f}%</b>", showarrow=False,
                xanchor='left', xshift=8,
                font=dict(color=colors[col], size=13))
        fig.update_layout(
            template='plotly_dark', height=380,
            margin=dict(l=40, r=70, t=48, b=30),
            title=f"YTD 성과 ({start} ~ · 두 북 10% 볼 정규화 · 백테스트 로그 기준)",
            yaxis_title="누적 PnL (%, 배정 자본 기준)",
            paper_bgcolor='#13161f', plot_bgcolor='#13161f',
            legend=dict(orientation='h', y=1.0, x=0))
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"⚠️ YTD 성과 차트 생성 실패 (백테스트 로그 필요): {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 오늘의 공식 트레이딩 시그널 (main.py --mode signals 와 동일 경로)
# ─────────────────────────────────────────────────────────────────────────────
def build_signal_table(selector, prices, max_corr, tradeable):
    """selector.get_trading_report 로 공식 집계 시그널을 계산해 자산별 행으로.

    main.py --mode signals 와 같은 파이프라인: 상관필터 → Sharpe 가중(앙상블
    축소 포함) → 금리 alpha 부스트 → 인버스볼 사이징.
    """
    report = selector.get_trading_report(prices, max_correlation=max_corr)
    agg = {r['asset']: r for r in report.get('aggregated_positions', [])}
    rows = []
    for asset in sorted(tradeable,
                        key=lambda a: (CLASS_ORDER.get(classify_asset_class(a), 9), a)):
        r = agg.get(asset)
        if r is None:
            rows.append({'asset': asset, 'klass': classify_asset_class(asset),
                         'dir': '-', 'conf': 0.0, 'pos': 0.0, 'n': 0})
            continue
        raw = float(r.get('raw_position', 0.0))
        rows.append({
            'asset': asset, 'klass': classify_asset_class(asset),
            'dir': 'LONG' if raw > 1e-9 else ('SHORT' if raw < -1e-9 else '-'),
            'conf': abs(raw),
            'pos': float(r.get('position', 0.0)),
            'n': int(r.get('num_strategies', 0)),
        })
    return rows, report.get('total_active_strategies', 0)


def print_signal_table(sig_rows, n_active, max_corr, signal_date, delta_info=None):
    print(f"\n▌ 오늘의 트레이딩 시그널  (기준일 {signal_date} · 금리=슬리브 엔진 · "
          f"FX=전략 공장(상관필터 {max_corr}) — main.py --mode signals 동일)")
    print(f"  공장 선택 전략 {n_active}개")
    print(f"  {'자산':<26} {'기초지표':>14} {'가격':>9} {'방향':<8} {'확신도':>6} "
          f"{'포지션':>7} {'Δ전일만원':>9} {'델타만원':>8} {'출처':>6}")
    last_klass = None
    for r in sig_rows:
        if r['klass'] != last_klass:
            print(f"  ── {CLASS_LABEL.get(r['klass'], r['klass'])} ──")
            last_klass = r['klass']
        arrow = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '· 중립'}[r['dir']]
        src = 'SLV' if r.get('src') == 'sleeve' else f"전략{r['n']}"
        ddw = f"{r['ddelta_w']:>+9.0f}" if 'ddelta_w' in r else f"{'-':>9}"
        dw = f"{r['delta_w']:>+8.0f}" if 'delta_w' in r else f"{'-':>8}"
        px = _px_fmt(r.get('price'))
        und = r.get('underlying', '-')
        print(f"  {asset_label(r['asset']):<26} {und:>14} {px:>9} {arrow:<8} {r['conf']:>6.2f} "
              f"{r['pos']:>+7.2f} {ddw} {dw} {src:>6}")
    if delta_info:
        d = delta_info
        g_txt = (f" · 그로스 {d['gross_w']:,.0f}만원 / 한도 {d['gross_budget']:,.0f}만원 "
                 f"(사용률 {d['gross_usage']:.0%})" if d.get('gross_budget')
                 else f" · 그로스 {d['gross_w']:,.0f}만원")
        print(f"\n  💰 금리 북 순델타 {d['net_w']:+,.0f}만원 / 한도 ±{d['budget']:,.0f}만원 "
              f"(사용률 {d['usage']:.0%}){g_txt}")
        print(f"     환산: 포지션 1.0 = {d['per_unit']:,.0f}만원 — 과거 최대 "
              f"|순포지션| {d['max_net_units']:.2f} / 그로스 {d['max_gross_units']:.2f} 중 "
              f"{'그로스' if d['binding'] == 'gross' else '순델타'} 한도가 결정")


# ─────────────────────────────────────────────────────────────────────────────
# 콘솔 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_console(rows, factory_dir, signal_date, official_dir=None):
    rows = sorted(rows, key=sort_key)
    c = counts(rows)
    total = len(rows)

    print("\n" + "=" * 92)
    print(f"  전략 공장 대시보드  |  폴더: {factory_dir.name}  |  기준일: {signal_date}")
    print(f"  탐색 공간 {total:,}개 조합  |  이번 달 활성 {c[ST_ENTER] + c[ST_ACTIVE]}개"
          + (f" (저장만 {c[ST_STORED]}개)" if c[ST_STORED] else "")
          + f"  |  오늘 진입 {c[ST_ENTER]}개")
    print("=" * 92)

    by_strat, strats_sorted = group_by_strat(rows)

    last_type = None
    for strat_name in strats_sorted:
        grp = by_strat[strat_name]
        strat_type = grp[0]['type']
        if strat_type != last_type:
            print(f"\n▒▒▒ {strat_type.upper()} ▒▒▒")
            last_type = strat_type
        gc = counts(grp)
        n_act = gc[ST_ENTER] + gc[ST_ACTIVE]
        print(f"  ── {strat_name:<24} 후보 {len(grp):>3}개 · 활성 {n_act:>2}개 · 진입 {gc[ST_ENTER]:>2}개")
        for r in grp:
            if r['status'] == ST_CAND:
                continue
            mark = ST_LABEL[r['status']] + ('✓' if r.get('picked') else ' ')
            arrow = {'LONG': '▲L', 'SHORT': '▼S', '-': '  '}[r['dir']]
            sh6 = f"{r['sharpe_6m']:.2f}" if r['sharpe_6m'] is not None else "  - "
            print(f"       [{mark}] {arrow}  {short_name(r['asset']):<12} Sh6m {sh6:>5}  | {r['desc']}")

    print("\n" + "=" * 92)


# ─────────────────────────────────────────────────────────────────────────────
# HTML 출력 — 모든 조합 전체 나열 (자산별 접이식)
# ─────────────────────────────────────────────────────────────────────────────
def _spark_svg(vals, w=150, h=30, guides=()):
    """인라인 SVG 스파크라인 (0선 + 옵션 가이드라인, 마지막 값 부호로 색상)."""
    vals = [float(v) for v in vals if v == v]
    if len(vals) < 2:
        return ''
    mn = min(min(vals), 0.0, *[g for g in guides] or [0.0])
    mx = max(max(vals), 0.0, *[g for g in guides] or [0.0])
    rng = (mx - mn) or 1.0
    pts = " ".join(
        f"{i * w / (len(vals) - 1):.1f},{h - (v - mn) / rng * h:.1f}"
        for i, v in enumerate(vals))
    zy = h - (0.0 - mn) / rng * h
    color = '#34d399' if vals[-1] > 0 else ('#f87171' if vals[-1] < 0 else '#9aa0aa')
    extra = "".join(
        f"<line x1='0' y1='{h - (g - mn) / rng * h:.1f}' x2='{w}' "
        f"y2='{h - (g - mn) / rng * h:.1f}' stroke='#666' stroke-dasharray='3,3' stroke-width='0.8'/>"
        for g in guides)
    return (f"<svg width='{w}' height='{h}' style='vertical-align:middle;'>"
            f"<line x1='0' y1='{zy:.1f}' x2='{w}' y2='{zy:.1f}' stroke='#3a3f4d' stroke-width='1'/>"
            f"{extra}<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.4'/></svg>")


def write_html(rows, factory_dir, signal_date, out_path,
               sig_rows=None, n_active_sel=0, max_corr=0.5, official_dir=None,
               sleeve_snap=None, delta_info=None, perf_html=None, fx_hist=None):
    rows = sorted(rows, key=sort_key)
    c = counts(rows)
    total = len(rows)

    by_strat, strats_sorted = group_by_strat(rows)

    css = """
    body{font-family:'Malgun Gothic',system-ui,sans-serif;background:#0f1116;color:#e6e6e6;margin:0;padding:24px;}
    h1{font-size:22px;margin:0 0 4px;font-weight:700;letter-spacing:-0.5px;} 
    h2{font-size:16px;margin:24px 0 8px;color:#8ab4f8;border-bottom:1px solid #2a2f3a;padding-bottom:6px;}
    .meta{color:#9aa0aa;font-size:13px;margin-bottom:18px;}
    .klass{margin:26px 0 8px;font-size:15px;color:#c4b5fd;border-bottom:1px solid #2a2f3a;padding-bottom:4px;font-weight:600;}
    details{margin:8px 0;background:#13161f;border-radius:6px;border:1px solid #1e2330;overflow:hidden;} 
    summary{cursor:pointer;font-weight:600;font-size:14px;padding:10px 14px;background:#171b24;transition:background 0.2s;}
    summary:hover{background:#1e2330;} summary .net{font-weight:400;color:#9aa0aa;margin-left:8px;}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:0;}
    td,th{padding:8px 12px;text-align:left;border-bottom:1px solid #1d2230;}
    th{color:#9aa0aa;font-weight:500;background:#171b24;position:sticky;top:0;}
    .pill{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;text-align:center;}
    .st0{background:#10381f;color:#34d399;} .st1{background:#16243f;color:#8ab4f8;}
    .st2{background:#3a2e12;color:#fbbf24;} .st3{background:#1c1f27;color:#5d6470;}
    .st_on{background:#10381f;color:#34d399;} .st_off{background:#1c1f27;color:#7a828e;} .st_sim{background:#3a2e12;color:#fbbf24;}
    tr.cand td{color:#5d6470;} tr.cand .pill{font-weight:400;}
    .long{color:#34d399;} .short{color:#f87171;} .flat{color:#555c6b;}
    .type{color:#c4b5fd;} .sh{color:#e6e6e6;font-variant-numeric:tabular-nums;font-weight:500;}
    .desc{color:#cbd2da;} tr.cand .desc, tr.cand .type{color:#5d6470;}
    .params{color:#7a828e;font-size:11.5px;}
    hr{border:0;border-top:1px solid #2a2f3a;margin:24px 0;}
    table.strats td:not(.desc), table.strats th{white-space:nowrap;}
    """
    st_pill = {ST_ENTER: ("st0", "진입"), ST_ACTIVE: ("st1", "활성"),
               ST_STORED: ("st2", "저장"), ST_CAND: ("st3", "후보")}

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>전략 공장 대시보드</title><style>{css}</style></head><body>"]
    parts.append("<h1>전략 공장 대시보드</h1>")

    # 1. 퀀트 전략 리스트 — 기존 운용 전략(고정) + 프로덕션 두 북의 한눈 표:
    #    공장 FX 전략은 enabled_cells 탐색 공간 포함 여부로 ON/OFF,
    #    슬리브 금리 팩터 전략은 sleeve_weights > 0 여부로 ON/OFF.
    parts.append("<h2>[퀀트 전략 리스트]</h2>")
    parts.append("<table class='strats' style='max-width:1200px;margin-bottom:20px;border:1px solid #1e2330;border-radius:6px;overflow:hidden;'>")
    parts.append("<tr style='background:#171b24;'><th>No.</th><th>이름</th><th>자산</th><th>설명</th><th>오버나잇</th><th>상태</th></tr>")
    
    idx = 1
    # 기존 운용 전략
    parts.append("<tr style='background:#1b1e27;'><td colspan='6' style='font-weight:bold;color:#fbbf24;font-size:12.5px;'>기존 운용 전략</td></tr>")
    for s in STATIC_STRATEGIES:
        s_class = "st_on" if s["status"] == "ON" else ("st_off" if s["status"] == "OFF" else "st_sim")
        parts.append(f"<tr>"
                     f"<td style='text-align:center;color:#9aa0aa;'>{idx}</td>"
                     f"<td style='font-weight:bold;color:#8ab4f8;'>{html.escape(s['name'])}</td>"
                     f"<td>{html.escape(s['asset'])}</td>"
                     f"<td class='desc'>{html.escape(s['desc'])}</td>"
                     f"<td style='text-align:center;'>{html.escape(s['overnight'])}</td>"
                     f"<td><span class='pill {s_class}'>{html.escape(s['status'])}</span></td>"
                     f"</tr>")
        idx += 1

    # 전략 공장 — FX 북: 현재 탐색 공간(enabled_cells)에 있는 전략만 ON
    universe_names = {r['name'] for r in rows}
    parts.append("<tr style='background:#1b1e27;'><td colspan='6' style='font-weight:bold;color:#34d399;font-size:12.5px;'>전략 공장 — FX 북 (2016+ 검증 SR 0.94)</td></tr>")
    fac_sorted = sorted(QUANT_STRATS_INFO.items(),
                        key=lambda kv: kv[0] not in universe_names)  # ON 먼저
    for key, info in fac_sorted:
        on = key in universe_names
        status, s_class = ('ON', 'st_on') if on else ('OFF', 'st_off')
        asset_txt = 'FX' if on else info['asset']
        parts.append(f"<tr>"
                     f"<td style='text-align:center;color:#9aa0aa;'>{idx}</td>"
                     f"<td style='font-weight:bold;color:#8ab4f8;'>{html.escape(info['name'])}</td>"
                     f"<td>{html.escape(asset_txt)}</td>"
                     f"<td class='desc'>{html.escape(info['desc'])}</td>"
                     f"<td style='text-align:center;'>{html.escape(info['overnight'])}</td>"
                     f"<td><span class='pill {s_class}'>{html.escape(status)}</span></td>"
                     f"</tr>")
        idx += 1

    # 슬리브 엔진 — 금리 북: 가중치 > 0 인 슬리브(스냅샷에 존재)만 ON
    active_sleeves = set((sleeve_snap or {}).get('sleeves', {}).keys())
    parts.append("<tr style='background:#1b1e27;'><td colspan='6' style='font-weight:bold;color:#c4b5fd;font-size:12.5px;'>슬리브 엔진 — 금리 북 (2016+ 검증 SR 0.97 · 통합 북 1.39)</td></tr>")
    for key, name, desc in SLEEVE_STRATS:
        on = key in active_sleeves
        status, s_class = ('ON', 'st_on') if on else ('OFF', 'st_off')
        parts.append(f"<tr>"
                     f"<td style='text-align:center;color:#9aa0aa;'>{idx}</td>"
                     f"<td style='font-weight:bold;color:#8ab4f8;'>{html.escape(name)}</td>"
                     f"<td>글로벌 금리</td>"
                     f"<td class='desc'>{html.escape(desc)}</td>"
                     f"<td style='text-align:center;'>O</td>"
                     f"<td><span class='pill {s_class}'>{html.escape(status)}</span></td>"
                     f"</tr>")
        idx += 1
    parts.append("</table>")

    # 2. 오늘의 트레이딩 시그널 (자산별 시그널)
    if sig_rows:
        parts.append("<h2>[자산별 트레이딩 시그널]</h2>")
        parts.append("<div class='meta'>포지션 = 기준자본 대비 <b>명목 배수</b>. "
                     "인버스볼 사이징이 반영돼 저변동 자산이 큰 숫자를 받음 — "
                     "명목이 커도 리스크 기여는 변동성에 비례. "
                     "금리 북 전체는 기준자본의 연 10% 변동성으로 타게팅. "
                     "120일 시계열: 금리=슬리브 엔진 실시간 / FX=공장 백테스트 로그 "
                     "(마지막 백테스트 실행 시점까지).</div>")
        if delta_info:
            d = delta_info
            net_cls = 'long' if d['net_w'] > 0 else ('short' if d['net_w'] < 0 else 'flat')
            big = "font-size:15px;font-weight:700;"
            g_txt = (f" · 그로스 <b style='{big}color:#e6e6e6;'>{d['gross_w']:,.0f}만원</b> / "
                     f"한도 {d['gross_budget']:,.0f}만원 "
                     f"(사용률 <b style='{big}color:#e6e6e6;'>{d['gross_usage']:.0%}</b>)"
                     if d.get('gross_budget') else f" · 그로스 {d['gross_w']:,.0f}만원")
            parts.append(
                f"<div class='meta'>💰 금리 북 순델타 "
                f"<b class='{net_cls}' style='{big}'>{d['net_w']:+,.0f}만원</b> / "
                f"한도 ±{d['budget']:,.0f}만원 "
                f"(사용률 <b class='{net_cls}' style='{big}'>{d['usage']:.0%}</b>){g_txt} · "
                f"환산 <b>포지션 1.0 = {d['per_unit']:,.0f}만원</b> "
                f"(과거 최대 |순| {d['max_net_units']:.2f} / 그로스 {d['max_gross_units']:.2f} 중 "
                f"{'그로스' if d['binding'] == 'gross' else '순델타'} 한도가 결정)</div>")
        sig_hist = (sleeve_snap or {}).get('history')
        parts.append("<table style='max-width:1060px;border:1px solid #1e2330;border-radius:6px;overflow:hidden;'>")
        parts.append("<tr style='background:#171b24;'><th>자산</th><th>기초 금리/환율</th><th>방향</th>"
                     "<th>확신도</th><th>포지션(변동성조정)</th><th>Δ전일(만원)</th><th>델타(만원)</th>"
                     "<th>출처</th><th>최근 120일간의 포지션 변동</th></tr>")
        last_k = None
        for r in sig_rows:
            if r['klass'] != last_k:
                parts.append(f"<tr><td colspan='9' class='klass' style='background:#13161f;padding:6px 12px;font-size:13px;'>"
                             f"{CLASS_LABEL.get(r['klass'], r['klass'])}</td></tr>")
                last_k = r['klass']
            dcls = {'LONG': 'long', 'SHORT': 'short', '-': 'flat'}[r['dir']]
            dtxt = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '· 중립'}[r['dir']]
            src = '슬리브' if r.get('src') == 'sleeve' else f"전략 {r['n']}개"
            if 'ddelta_w' in r:
                dp_cls = ('long' if r['ddelta_w'] > 0.5 else
                          ('short' if r['ddelta_w'] < -0.5 else 'flat'))
                dpos_td = (f"<td class='{dp_cls}' style='font-variant-numeric:tabular-nums;'>"
                           f"{r['ddelta_w']:+,.0f}</td>")
            else:
                dpos_td = "<td class='sh'>–</td>"
            dw = f"{r['delta_w']:+,.0f}" if 'delta_w' in r else '–'
            if r.get('src') == 'sleeve' and sig_hist is not None and r['asset'] in sig_hist.columns:
                spark = _spark_svg(sig_hist[r['asset']].tolist())
            elif r['klass'] == 'fx' and fx_hist is not None and r['asset'] in fx_hist.columns:
                # FX는 공장 백테스트 로그의 일별 포지션 (진입/청산 계단 패턴)
                spark = _spark_svg(fx_hist[r['asset']].tolist())
            else:
                spark = '–'
            px_span = (f" <span class='params'>{_px_fmt(r['price'])}</span>"
                       if r.get('price') is not None else "")
            und = html.escape(r.get('underlying', '–'))
            parts.append(f"<tr><td>{html.escape(asset_label(r['asset']))}{px_span}</td>"
                         f"<td class='sh'>{und}</td>"
                         f"<td class='{dcls}'>{dtxt}</td>"
                         f"<td class='sh'>{r['conf']:.2f}</td>"
                         f"<td class='sh'>{r['pos']:+.2f}</td>"
                         f"{dpos_td}"
                         f"<td class='sh'>{dw}</td>"
                         f"<td class='sh'>{src}</td>"
                         f"<td>{spark}</td></tr>")
        parts.append("</table>")

    # 2a. YTD 성과 (plotly: FX / RATES / COMBINED — combined_report 와 동일 정규화)
    if perf_html:
        parts.append("<h2>[YTD 성과 — FX · RATES · COMBINED]</h2>")
        parts.append(f"<div style='max-width:1060px;'>{perf_html}</div>")

    # 2b. 슬리브 엔진 (금리 북): 슬리브별 → 자산별
    if sleeve_snap:
        d = sleeve_snap['date'].date() if hasattr(sleeve_snap['date'], 'date') else sleeve_snap['date']
        s_assets = sorted(sleeve_snap['target'])
        parts.append(f"<h2>[슬리브 엔진 — 금리 북]</h2>"
                     f"<div class='meta'>기준일 {d} · 연속 시그널 → 인버스볼 × 포트 볼타게팅 × 스무딩 · "
                     f"main.py --mode signals 금리 SLEEVE 행과 동일</div>")
        for name in SLEEVE_ORDER:
            if name not in sleeve_snap['sleeves']:
                continue
            s = sleeve_snap['sleeves'][name]
            title, desc = SLEEVE_INFO.get(name, (name, ''))
            parts.append(f"<details open><summary>{html.escape(title)}"
                         f"<span class='net'>가중 {s['weight']:.1f} · {html.escape(desc)}</span></summary>")
            parts.append("<table><tr style='background:#171b24;'><th>자산</th><th>방향</th><th>시그널(z)</th></tr>")
            for a in s_assets:
                v = s['signals'].get(a)
                if v is None:
                    continue
                dr = _sleeve_dir(v, 0.1)
                dcls = {'LONG': 'long', 'SHORT': 'short', '-': 'flat'}[dr]
                dtxt = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '· 중립'}[dr]
                parts.append(f"<tr><td>{html.escape(asset_label(a))}</td>"
                             f"<td class='{dcls}'>{dtxt}</td><td class='sh'>{v:+.2f}</td></tr>")
            parts.append("</table></details>")
        # 순델타 시계열 차트 (최근 250일, 한도 가이드라인)
        net_hist = sleeve_snap.get('net_hist')
        if net_hist is not None and len(net_hist) > 2 and delta_info:
            _pu = delta_info['per_unit']
            net_w_series = (net_hist.tail(250) * _pu).tolist()
            guides = [delta_info['budget'], -delta_info['budget']]
            parts.append("<details open><summary>금리 북 순델타 추이 (최근 250일, 만원)"
                         f"<span class='net'>점선 = 한도 ±{delta_info['budget']:,.0f}만원 · "
                         f"현재 {delta_info['net_w']:+,.0f}만원</span></summary>"
                         f"<div style='padding:12px 14px;'>{_spark_svg(net_w_series, w=860, h=90, guides=guides)}</div>"
                         "</details>")

        hist = sleeve_snap.get('history')
        prev_map = sleeve_snap.get('prev', {})
        per_unit = delta_info['per_unit'] if delta_info else None
        parts.append("<details open><summary>최종 목표 포지션"
                     "<span class='net'>주문 기준 — 위 슬리브들의 가중 합성 후 리스크 사이징</span></summary>")
        parts.append("<table><tr style='background:#171b24;'><th>자산</th><th>방향</th>"
                     "<th>목표 포지션</th><th>Δ전일</th><th>델타(만원)</th>"
                     "<th>최근 120일간의 포지션 변동</th></tr>")
        for a in s_assets:
            p = sleeve_snap['target'][a]
            q = prev_map.get(a, p)
            dr = _sleeve_dir(p)
            dcls = {'LONG': 'long', 'SHORT': 'short', '-': 'flat'}[dr]
            dtxt = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '· 중립'}[dr]
            dp = p - q
            dp_cls = 'long' if dp > 0.005 else ('short' if dp < -0.005 else 'flat')
            dw = f"{p * per_unit:+,.0f}" if per_unit else '–'
            spark = _spark_svg(hist[a].tolist()) if (hist is not None and a in hist.columns) else ''
            parts.append(f"<tr><td>{html.escape(asset_label(a))}</td>"
                         f"<td class='{dcls}'>{dtxt}</td><td class='sh'>{p:+.2f}</td>"
                         f"<td class='{dp_cls}' style='font-variant-numeric:tabular-nums;'>{dp:+.2f}</td>"
                         f"<td class='sh'>{dw}</td><td>{spark}</td></tr>")
        parts.append("</table></details>")

    parts.append("<hr>")

    # 3. 상세 전략 목록 (전략별 접이식 분류)
    parts.append(f"<div class='meta'>폴더 {html.escape(factory_dir.name)} · 기준일 {signal_date} · "
                 f"탐색 공간 <b>{total:,}개</b> 조합 · 이번 달 활성 "
                 f"<b class='st1'>{c[ST_ENTER] + c[ST_ACTIVE]}개</b>"
                 + (f" · 저장만 {c[ST_STORED]}개" if c[ST_STORED] else "")
                 + f" · 오늘 진입 <b class='st0'>{c[ST_ENTER]}개</b></div>")

    TYPE_LABEL = {
        'momentum': '모멘텀 전략 (Momentum)',
        'mean_reversion': '평균회귀 전략 (Mean Reversion)',
        'advanced': '고급/다자산 전략 (Advanced)',
        'alpha1': '알파1 전략 (Alpha1)',
        'alpha2': '알파2 전략 (Alpha2)',
        'alpha4': '알파4 전략 (Alpha4)'
    }

    last_type = None
    for strat_name in strats_sorted:
        grp = by_strat[strat_name]
        strat_type = grp[0]['type']
        if strat_type != last_type:
            parts.append(f"<div class='klass'>{TYPE_LABEL.get(strat_type, strat_type.upper())}</div>")
            last_type = strat_type
        gc = counts(grp)
        n_act = gc[ST_ENTER] + gc[ST_ACTIVE]
        open_attr = " open" if n_act else ""

        parts.append(f"<details{open_attr}><summary>{html.escape(strat_name)}"
                     f"<span class='net'>후보 {len(grp)}개 · 활성 {n_act}개 · "
                     f"진입 {gc[ST_ENTER]}개</span></summary>")
        parts.append("<table><tr style='background:#171b24;'><th>상태</th><th>방향</th><th>자산</th>"
                     "<th>파라미터</th><th>Sh3Y</th><th>Sh6M</th><th>설명</th></tr>")
        
        # 자산군 -> 자산명 -> 상태 순 정렬
        grp_sorted = sorted(grp, key=lambda r: (CLASS_ORDER.get(r['klass'], 9), r['asset'], r['status']))
        for r in grp_sorted:
            cls, txt = st_pill[r['status']]
            if r.get('picked'):
                txt += ' ✓'
            tr_cls = " class='cand'" if r['status'] == ST_CAND else ""
            dcls = {'LONG': 'long', 'SHORT': 'short', '-': 'flat'}[r['dir']]
            dtxt = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '·'}[r['dir']]
            sh3 = f"{r['sharpe_3y']:.2f}" if r['sharpe_3y'] is not None else "–"
            sh6 = f"{r['sharpe_6m']:.2f}" if r['sharpe_6m'] is not None else "–"
            parts.append(
                f"<tr{tr_cls}><td><span class='pill {cls}'>{txt}</span></td>"
                f"<td class='{dcls}'>{dtxt}</td>"
                f"<td style='font-weight:500;'>{html.escape(asset_label(r['asset']))}</td>"
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
    ap.add_argument('--max-corr', type=float, default=1.0,
                    help="공식 시그널 집계의 상관 필터 (1.0=필터 없음, 2026-06-11 "
                         "정직 A/B 검증 구성. 백테스트 기본과 동일)")
    ap.add_argument('--delta-budget', type=float, default=5000.0,
                    help="금리 북 순델타 한도 (만원, 기본 5000)")
    ap.add_argument('--gross-budget', type=float, default=8000.0,
                    help="금리 북 그로스 한도 (만원, 기본 8000). 순/그로스 중 "
                         "더 빡빡한 쪽이 '포지션 1.0 = N만원' 환산 계수를 결정")
    ap.add_argument('--perf-start', default='2026-01-01',
                    help="HTML YTD 성과 차트 시작일 (기본 2026-01-01)")
    ap.add_argument('--html', action='store_true', help="HTML 대시보드도 저장")
    args = ap.parse_args()

    factory_base = Path(__file__).parent.parent / 'src' / 'factory'
    factory_dir = resolve_factory_dir(factory_base, args.date)

    # 데이터 로드
    print("📊 가격 데이터 로딩...")
    loader = DataLoader()
    prices_raw = loader.load_data(use_cache=True)
    prices = DataPreprocessor(prices_raw).clean().get_data()
    prices = loader.merge_signal_yields(prices)  # alpha4 신호용 금리 컬럼

    factory = StrategyFactory(storage_dir=str(factory_dir))
    selector = StrategySelector(factory=factory)

    rows = build_universe_rows(prices, loader, factory, selector,
                               target_date=args.signal_date)

    # 슬리브 엔진 금리 북 스냅샷 (공장과 별개 — 2010+ 자체 패널)
    sleeve_snap = build_sleeve_snapshot(loader)
    if args.asset:
        rows = [r for r in rows if args.asset.lower() in r['asset'].lower()]
        if sleeve_snap:
            t = {a: p for a, p in sleeve_snap['target'].items()
                 if args.asset.lower() in a.lower()}
            if t:
                sleeve_snap = {**sleeve_snap, 'target': t,
                               'sleeves': {n: {**s, 'signals': {a: v for a, v in s['signals'].items() if a in t}}
                                           for n, s in sleeve_snap['sleeves'].items()}}
            else:
                sleeve_snap = None
    if not rows and not sleeve_snap:
        print("⚠️ 표시할 전략 조합이 없습니다.")
        factory.close()
        return

    signal_date = max((r['date'] for r in rows if r['date']), default='N/A')
    if signal_date == 'N/A' and sleeve_snap:
        signal_date = str(sleeve_snap['date'].date())

    # 공식 집계 시그널 (자산 단위 최종 포지션): FX=공장, 금리=슬리브
    tradeable = sorted({r['asset'] for r in rows})
    sig_rows, n_active_sel = build_signal_table(selector, prices, args.max_corr, tradeable)
    delta_info = None
    if sleeve_snap:
        delta_info = compute_delta_info(sleeve_snap, args.delta_budget,
                                        gross_budget_w=args.gross_budget)
        per_unit = delta_info['per_unit'] if delta_info else None
        sig_rows = sleeve_signal_rows(sleeve_snap, per_unit) + sig_rows
        sig_rows.sort(key=lambda r: (CLASS_ORDER.get(r['klass'], 9), r['asset']))
    # 현재가 + 기초지표(일드/통상 환율 호가) 부착 (최신 거래일 종가)
    last_px = prices.iloc[-1]
    for r in sig_rows:
        if r['asset'] in last_px.index:
            r['price'] = float(last_px[r['asset']])
    attach_underlying(sig_rows, prices)
    print_signal_table(sig_rows, n_active_sel, args.max_corr, signal_date,
                       delta_info=delta_info)
    if sleeve_snap:
        print_sleeve_console(sleeve_snap)

    # 상관필터가 채택한 전략 태깅 (공식 시그널은 이 전략들로만 계산됨)
    sel_keys = {combo_key(s.get('strategy_name'), s.get('asset'),
                          s.get('related_asset'), s.get('params'))
                for s in selector.active_strategies}
    for r in rows:
        r['picked'] = combo_key(r['name'], r['asset'],
                                r.get('related'), r['params']) in sel_keys
    official_dir = {r['asset']: r['dir'] for r in sig_rows}

    print_console(rows, factory_dir, signal_date, official_dir=official_dir)

    if args.html:
        from datetime import date
        out = Path(f"strategy_dashboard_{date.today().isoformat()}.html")
        perf_html = build_ytd_perf_html(start=args.perf_start)
        write_html(rows, factory_dir, signal_date, out,
                   sig_rows=sig_rows, n_active_sel=n_active_sel, max_corr=args.max_corr,
                   official_dir=official_dir, sleeve_snap=sleeve_snap,
                   delta_info=delta_info, perf_html=perf_html,
                   fx_hist=load_fx_pos_history())

    factory.close()


if __name__ == '__main__':
    main()
