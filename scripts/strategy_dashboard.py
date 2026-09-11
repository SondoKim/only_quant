"""
금리 북 대시보드 (SleeveEngine)
================================

2026-09-11 FX 전략 공장 폐기(git tag `pre-fx-factory-retire`) 이후 이 스크립트는
금리 북(슬리브 엔진) 전용이다. FX 는 매매하지 않고, 슬리브 엔진의 FX 팩터 신호를
'모니터링' 섹션에만 띄운다 (주문 후보 표에는 주문할 수 있는 것만).

콘솔: 주문 후보 표(韓美 4종) → 금리 북 자동 해석 → 슬리브별 z → FX 모니터링
HTML : [퀀트 전략 리스트] · [자산별 트레이딩 시그널] · [YTD 성과 — 금리 북] ·
       [슬리브 엔진 — 금리 북] · [FX 팩터 모니터링 — 매매 안 함]

⚠ total_dashboard(Streamlit, D:/김선도/Python/total_dashboard) 가 이 HTML 을
파싱한다 — 아래 계약을 바꾸지 말 것:
  · [퀀트 전략 리스트] 제목 뒤 첫 <table>: 헤더 No./이름/자산/설명/오버나잇/상태,
    colspan 그룹행 키워드 '기존 운용 전략' / '전략 공장' / '슬리브 엔진'.
  · '방향' + '오늘 포지션' 두 컬럼을 동시에 가진 표 = 시그널 표. 다른 표는 이 조합을
    쓰지 않는다 (FX 모니터링 표는 '가상 포지션').
  · <!--SLEEVE_NARRATIVE_B64:...--> 주석 = 자동 해석 페이로드 (표와 같은 파일).
  · 산출물은 reports/dashboards/strategy_dashboard_{오늘}.html, 최신 1개만 유지.

사용 예:
    python scripts/strategy_dashboard.py            # 콘솔 (= main.py --mode signals)
    python scripts/strategy_dashboard.py --html     # HTML 도 저장
    python scripts/strategy_dashboard.py --asset TU1
"""

import argparse
import base64
import json
import sys
import html
import textwrap
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:  # 콘솔 한글/기호 깨짐 방지
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

ROOT = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# 자산 친화명 · 자산군
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


def classify_asset_class(ticker: str) -> str:
    """티커 → 자산군 (옛 selector.classify_asset_class 대체)."""
    if 'Comdty' in ticker and 'NQ' not in ticker:
        return 'rates'
    if 'Curncy' in ticker:
        return 'fx'
    if 'Index' in ticker:
        return 'index'
    return 'other'


def asset_label(ticker: str) -> str:
    name = ASSET_NAMES.get(ticker)
    return f"{ticker} ({name})" if name else ticker


def short_name(ticker: str) -> str:
    return ASSET_NAMES.get(ticker, ticker)


# ─────────────────────────────────────────────────────────────────────────────
# [퀀트 전략 리스트] 고정 항목
# ─────────────────────────────────────────────────────────────────────────────
STATIC_STRATEGIES = [
    {"name": "리버전", "asset": "3선, 10선", "desc": "전일 미국장 금리방향과 반대로 진입", "overnight": "X", "status": "ON"},
    {"name": "채널", "asset": "KRW, JPY", "desc": "FX 가격 밴드 스윙 트레이딩", "overnight": "O", "status": "ON"},
    {"name": "PCA금리", "asset": "미국, 독일금리", "desc": "이론금리 괴리분에 대한 회귀 가정 매매", "overnight": "O", "status": "모의"},
    {"name": "아이언콘돌", "asset": "미국금리옵션", "desc": "금리 옵션 양매도+양매수", "overnight": "O", "status": "모의"},
    {"name": "변동성 돌파 전략", "asset": "미국금리, JPY, G", "desc": "전일 변동성 range 당일 돌파 시 매매", "overnight": "X", "status": "OFF"},
    {"name": "FX캐리극대화 전략", "asset": "글로벌 FX", "desc": "캐리 극대화 fx long-short", "overnight": "O", "status": "OFF"}
]

# 폐기된 FX 전략 공장 — 전략 리스트에 기록용 한 줄로만 남긴다 (그룹 키워드 '전략 공장').
RETIRED_FACTORY = {
    "name": "FX 전략 공장 (조합 탐색 · 월별 재선정)", "asset": "FX",
    "desc": "2026-09-11 폐기 — 과최적화된 전략 조합의 관성을 매매하는 구조라 시장·퀀트 논리가 "
            "약하다고 판단 (2025 SR −1.03, 2026 YTD −0.59). 코드·DB 는 git tag "
            "pre-fx-factory-retire 에 보존",
    "overnight": "O", "status": "OFF",
}

# ─────────────────────────────────────────────────────────────────────────────
# 슬리브 엔진 (금리 북) — 슬리브별 → 자산별 현황
# ─────────────────────────────────────────────────────────────────────────────
SLEEVE_INFO = {
    'trend':  ('추세 (TSMOM)',     '6/12개월 가격 추세 z-score · 방향성'),
    'value':  ('밸류 (평균회귀)',  '2년 평균 대비 가격 괴리 — 과열 숏 / 과매도 롱 · 시장중립'),
    'carry':  ('캐리 (일드 레벨)', '환헤지 후 국가간 일드 레벨 z — 고금리국 롱 / 저금리국 숏 · 시장중립'),
    'curve':  ('커브 캐리',        '10Y−2Y 기울기 z — 스팁 롱 / 역전 숏 · 방향성 (미채택, 가중 0)'),
    'policy': ('정책 모멘텀',      '2Y 금리 6개월 변화 — 인하 사이클 롱 / 인상 사이클 숏 · 방향성'),
}
SLEEVE_ORDER = ['trend', 'value', 'carry', 'curve', 'policy']

# 한눈 표용 슬리브 전략명 (팩터별 전략 이름 + 설명)
SLEEVE_STRATS = [
    ('trend',  '글로벌 금리 추세 전략 (Sleeve)',
     '6/12개월 가격 추세 z-score 추종 (방향성)'),
    ('value',  '글로벌 금리 밸류 전략 (Sleeve)',
     '2년 평균 대비 가격 괴리 평균회귀 — 과열 숏 / 과매도 롱 (시장중립)'),
    ('carry',  '글로벌 금리 캐리 전략 (Sleeve)',
     '환헤지 후 국가간 일드 레벨 비교 — 고금리국 롱 / 저금리국 숏 (시장중립)'),
    ('curve',  '커브 캐리 전략 (Sleeve)',
     '자국 10Y−2Y 기울기 z — 스팁 롱 / 역전 숏 (A/B 결과 미채택)'),
    ('policy', '통화정책 사이클 전략 (Sleeve)',
     '2Y 금리 6개월 변화로 중앙은행 인하/인상 사이클 추종'),
]

# FX 모니터링 (매매 안 함) — 슬리브 엔진 FX 북의 팩터 설명
FX_SLEEVE_INFO = {
    'trend': ('추세', '6/12개월 통화 가격 추세 z · 방향성'),
    'value': ('밸류', '2년 평균 대비 괴리 · 시장중립'),
    'carry': ('캐리', '2Y 금리차(외국−미국) z · 시장중립'),
}


def build_sleeve_snapshots(loader):
    """엔진 1회 구성 → (금리 스냅샷, FX 모니터링 스냅샷, 일드 병합 가격 패널).

    2010+ 자체 패널(value 504d·trend 252d 워밍업), position_smooth·북스톱 적용 —
    옛 main._merge_sleeve_rates 와 동일 구성. 실패 시 (None, None, None).
    """
    try:
        import yaml as _yaml
        from src.sleeves.sleeve_engine import SleeveEngine
        cfg_path = ROOT / 'config' / 'indicators.yaml'
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = (_yaml.safe_load(f) or {}).get('sleeves', {}) or {}
        px = DataPreprocessor(
            loader.load_data(start_date='2010-01-01', use_cache=True)
        ).clean().get_data()
        yields = loader.load_signal_yields(start_date='2010-01-01', use_cache=True)
        macro = loader.load_signal_macro(start_date='2010-01-01', use_cache=True)
        engine = SleeveEngine(px, config=cfg, yields=yields, macro=macro)
        rates_snap = engine.sleeve_snapshot('rates')
        fx_snap = engine.sleeve_snapshot('fx') if engine.fx_assets else None
        return rates_snap, fx_snap, loader.merge_signal_yields(px)
    except Exception as e:
        print(f"⚠️ 슬리브 스냅샷 생성 실패: {e}")
        return None, None, None


def _sleeve_dir(v: float, thresh: float = 0.02) -> str:
    return 'LONG' if v > thresh else ('SHORT' if v < -thresh else '-')


def sleeve_signal_rows(snap, delta_per_unit=None):
    """스냅샷 → 공식 시그널 테이블용 금리 행 (main.py --mode signals 와 동일).

    delta_per_unit: 포지션 1.0당 만원 환산 계수.

    ⚠ signal_only_assets(英·日·豪)는 여기서 **제외한다**. 이 표는 주문 후보
    목록이고, 주문할 수 없는 자산이 섞이면 혼선만 준다 (포지션 0 이 '중립 판단'
    으로 오독됨). 이들의 팩터 시계열은 sleeve_factor_signals.csv →
    '금리 팩터 시그널 시계열' 차트에 그대로 남아 맥락을 제공한다.
    """
    sig_only = set(snap.get('signal_only') or [])
    rows = []
    for a in sorted(snap['target'], key=lambda x: (CLASS_ORDER.get(classify_asset_class(x), 9), x)):
        if a in sig_only:
            continue
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


# ─────────────────────────────────────────────────────────────────────────────
# 금리 북 시그널 자동 해석 (표 아래 렌더)
# ─────────────────────────────────────────────────────────────────────────────
# 슬리브별 읽기: (표시명, 평균이 +일 때의 뜻, −일 때의 뜻)
SLEEVE_READ = {
    'trend':  ('추세', '6/12개월 가격이 상승 추세', '6/12개월 가격이 하락 추세'),
    'value':  ('밸류', '2년 평균 대비 싼 편', '2년 평균 대비 비싼 편'),
    'carry':  ('캐리', '환헤지 후 일드 레벨이 상대적으로 높음',
               '환헤지 후 일드 레벨이 상대적으로 낮음'),
    'curve':  ('커브', '커브가 가팔라 롤다운 유리', '커브가 눌려 롤다운 불리'),
    'policy': ('정책 모멘텀', '단기금리 하락 = 완화 사이클',
               '단기금리 상승 = 긴축 사이클'),
}
# 컨빅션 분해에서 제외 — 별도 서브북이라 sleeve_weights 분모를 공유하지 않는다
NON_CONVICTION_SLEEVES = {'reversion'}


def _fmt_assets(items, short_first=None):
    """[(ticker, value)] → '미국 2Y -0.77 · 미국 10Y -0.50' 형태.

    short_first=True 면 더 숏인 쪽부터, False 면 더 롱인 쪽부터 정렬한다
    (None = 입력 순서 유지). 문장의 방향과 나열 순서를 맞추기 위한 인자.
    """
    if short_first is not None:
        items = sorted(items, key=lambda kv: kv[1] if short_first else -kv[1])
    return ' · '.join(f"{short_name(a)} {v:+.2f}" for a, v in items)


def sleeve_narrative(snap, per_unit=None):
    """오늘의 금리 북 시그널을 사람이 읽는 문장으로 자동 해석.

    스냅샷의 슬리브 z값·목표 포지션·북스톱 상태만으로 생성한다 (하드코딩된
    시황 문구 없음). 반환: [(머리말, 본문), ...] — 콘솔/HTML 이 같은 내용을
    렌더하므로 두 경로가 어긋날 수 없다.

    ⚠ 컨빅션은 '방향의 근거'일 뿐 포지션 크기와 비례하지 않는다. 최종 크기는
    인버스볼 → 볼타겟 → 스무딩(0.8) → 북스톱 → 노출스케일을 거친 결과다.
    """
    if not snap:
        return []
    sig_only = set(snap.get('signal_only') or [])
    tgt = {a: p for a, p in (snap.get('target') or {}).items() if a not in sig_only}
    if not tgt:
        return []
    sleeves = snap.get('sleeves') or {}
    prev = snap.get('prev') or {}
    out = []

    # ── 공통 집계 ──────────────────────────────────────────────────────
    TH = 0.02
    longs = sorted([(a, p) for a, p in tgt.items() if p > TH], key=lambda x: -x[1])
    shorts = sorted([(a, p) for a, p in tgt.items() if p < -TH], key=lambda x: x[1])
    flats = [a for a, p in tgt.items() if abs(p) <= TH]
    net = sum(tgt.values())
    gross = sum(abs(p) for p in tgt.values())
    ratio = abs(net) / gross if gross > 1e-9 else 0.0
    bs = snap.get('book_stop')

    def _stop_line():
        """북스톱 상태 문장 (스톱 비활성이면 None)."""
        if not bs:
            return None
        dd, sc = bs['dd'], bs['scale']
        win = bs.get('dd_window') or 0
        wtxt = f"{win}일 롤링 고점" if win else "전기간 고점"
        if sc >= 0.999:
            return ("북스톱",
                    f"정상 작동 중입니다. 섀도우 북 드로다운 {dd:.1f}% "
                    f"({wtxt} 대비)로 감축 임계 {bs['dd_half']:.0f}% 아래 — "
                    f"포지션 100% 집행.")
        if sc > 0.01:
            return ("⚠ 북스톱",
                    f"감축 중입니다. 드로다운 {dd:.1f}% 가 {bs['dd_half']:.0f}% 를 "
                    f"넘어 금리 포지션이 ×{sc:g} 로 축소됐습니다. 섀도우 북이 "
                    f"회복하면 자동 복원되니 수동으로 되돌리지 마십시오.")
        return ("⛔ 북스톱",
                f"발동했습니다. 드로다운 {dd:.1f}% 가 플랫 임계 "
                f"{bs['dd_flat']:.0f}% 를 초과해 금리 북이 플랫(×0)입니다. "
                f"섀도우 북 회복 시 자동 재진입합니다.")

    # ── ⓪ 북이 비어 있으면 여기서 끝 (해석할 포지션이 없음) ─────────────
    # 이 분기가 없으면 그로스 0 인 북에 대해 "미국 숏(+0.00)이 한국보다 크다"
    # 같은 무의미한 문장이 만들어진다.
    if gross < 0.05:
        why = ("북스톱이 발동해 금리 포지션을 0으로 눌렀습니다."
               if bs and bs['scale'] < 0.01 else
               "팩터 신호가 서로 상쇄됐거나 변동성 워밍업 구간입니다 "
               "(볼타겟 초기 63거래일은 강제 플랫).")
        out.append(("포지션",
                    f"금리 북이 사실상 플랫입니다 (그로스 {gross:.2f}) — "
                    f"오늘 신규 주문 없음. {why}"))
        st = _stop_line()
        if st:
            out.append(st)
        return out

    # ── ① 지금 북이 어떤 포지션인가 ─────────────────────────────────────
    side = '롱' if net > 0 else '숏'

    if longs and not shorts:
        head = f"매매 {len(tgt)}종 전부 듀레이션 롱"
    elif shorts and not longs:
        head = f"매매 {len(tgt)}종 전부 듀레이션 숏"
    else:
        head = f"롱 {len(longs)}종 · 숏 {len(shorts)}종"
        if flats:
            head += f" · 중립 {len(flats)}종"
    delta_txt = (f" · 순델타 {net * per_unit:+,.0f}만원" if per_unit else "")
    out.append(("포지션",
                f"{head}입니다 (순 {net:+.2f} / 그로스 {gross:.2f} — "
                f"방향성 {ratio:.0%} / 국가간 상대가치 {1 - ratio:.0%}{delta_txt})."))

    # ── ② 팩터 분해 ────────────────────────────────────────────────────
    # 두 성분을 따로 잰다:
    #   contrib = 매매자산 평균 (= 북의 '방향성' 성분을 만든 기여도)
    #   gaps    = 국가 간 평균 격차 (= '상대가치' 성분을 만든 갈림)
    # 상대가치 위주 북에서는 평균이 상쇄돼 0 에 가까워지므로, 평균만 보고
    # "추세 +0.02 가 롱을 주도"라고 쓰면 사실을 왜곡한다 → ratio 로 분기.
    grp = {}
    for a, p in tgt.items():
        grp.setdefault(short_name(a).split()[0], []).append((a, p))

    w_map = snap.get('sleeve_weights') or {
        n: s.get('weight', 0.0) for n, s in sleeves.items()}
    denom = sum(abs(w) for n, w in w_map.items()
                if w and n not in NON_CONVICTION_SLEEVES) or 1.0
    contrib, gaps = {}, []
    for name, s in sleeves.items():
        if name in NON_CONVICTION_SLEEVES:
            continue
        w = s.get('weight', 0.0)
        sg = s.get('signals') or {}
        vals = [sg[a] for a in tgt if a in sg]
        if not w or not vals:
            continue
        contrib[name] = w * (sum(vals) / len(vals)) / denom
        gm = {g: [sg[a] for a, _ in its if a in sg] for g, its in grp.items()}
        gm = {g: sum(v) / len(v) for g, v in gm.items() if v}
        if len(gm) >= 2:
            gaps.append((max(gm.values()) - min(gm.values()), name, gm))

    def _label(name):
        return SLEEVE_READ.get(name, (name,))[0]

    def _kind(name):
        xs = (sleeves.get(name) or {}).get('xs_neutralize', 0.0)
        return '상대가치' if xs and xs >= 0.99 else '방향성'

    def _phr(name, c):
        label, pos_t, neg_t = SLEEVE_READ.get(name, (name, '양(+)', '음(−)'))
        return f"{label} {c:+.2f}({_kind(name)}) {pos_t if c > 0 else neg_t}"

    def _gm_txt(gm, short_first):
        return ' vs '.join(f"{g} {v:+.2f}" for g, v in sorted(
            gm.items(), key=lambda kv: kv[1] if short_first else -kv[1]))

    # 국가 평균이 이만큼(z) 벌어져야 '갈렸다'고 말한다 — 격차 0 인데
    # "갈림의 주 원인은 추세(미국 -0.40 vs 한국 -0.40)" 같은 문장을 막는다.
    GAP_TH = 0.10
    DIRECTIONAL = ratio >= 0.5
    if contrib and DIRECTIONAL:
        book_sign = 1.0 if side == '롱' else -1.0
        ranked = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
        drv = [(n, c) for n, c in ranked if c * book_sign > 0]
        opp = [(n, c) for n, c in ranked if c * book_sign <= 0]
        if drv:
            out.append(("근거", f"{side}을 주도하는 팩터 — "
                        + ' / '.join(_phr(n, c) for n, c in drv) + "."))
        if opp:
            opp_side = '숏' if side == '롱' else '롱'
            verdict = ("크기가 작아 방향을 뒤집지 못합니다"
                       if abs(opp[0][1]) < abs(drv[0][1]) * 0.6 and drv else
                       "크기가 비슷해 순노출이 그만큼 줄어듭니다")
            out.append(("반대편", f"{opp_side} 쪽을 가리키는 팩터 — "
                        + ' / '.join(_phr(n, c) for n, c in opp)
                        + f". 다만 {verdict}."))
    elif [g for g in gaps if g[0] >= GAP_TH]:
        top = sorted((g for g in gaps if g[0] >= GAP_TH), key=lambda x: -x[0])[:2]
        out.append(("근거",
                    f"방향성보다 국가간 상대가치가 큰 북입니다 (방향성 {ratio:.0%}) "
                    f"— 팩터 평균은 서로 상쇄되므로 국가를 가르는 힘으로 읽어야 "
                    f"합니다. 갈림이 큰 팩터: "
                    + ' / '.join(f"{_label(n)}({_kind(n)}) {_gm_txt(gm, False)}"
                                 for _, n, gm in top) + "."))
    elif contrib:
        out.append(("근거", "팩터 — " + ' / '.join(
            _phr(n, c) for n, c in
            sorted(contrib.items(), key=lambda kv: -abs(kv[1]))) + "."))

    # ── ③ 국가·만기별 차별화 ───────────────────────────────────────────
    if len(grp) >= 2:
        sums = {g: sum(v for _, v in its) for g, its in grp.items()}
        if min(sums.values()) < -1e-9 and max(sums.values()) > 1e-9:
            # 국가별로 방향이 갈린 경우 — '누가 더 크다'가 아니라 '어느 쪽이 롱/숏'
            segs = [f"{g}은 {'롱' if sums[g] > 0 else '숏'}"
                    f"({_fmt_assets(its, short_first=(sums[g] < 0))})"
                    for g, its in sorted(grp.items(), key=lambda kv: -sums[kv[0]])]
            txt = ' · '.join(segs) + " — 국가간 상대가치 포지션입니다."
        else:
            o = sorted(grp.items(), key=lambda kv: -abs(sums[kv[0]]))
            (bg, b_items), (sm, s_items) = o[0], o[-1]
            sf = (side == '숏')
            txt = (f"{bg} {side}({_fmt_assets(b_items, short_first=sf)})이 "
                   f"{sm}({_fmt_assets(s_items, short_first=sf)})보다 큽니다.")
        if gaps and DIRECTIONAL:      # RV 모드에선 위 '근거'와 중복되므로 생략
            gtop = max(gaps, key=lambda x: x[0])
            if gtop[0] >= GAP_TH:
                _, gname, gm = gtop
                txt += (f" 갈림의 주 원인은 {_label(gname)}입니다 "
                        f"({_gm_txt(gm, short_first=(side == '숏'))}).")
        out.append(("국가별", txt))

    # ── ④ 전일 대비 ────────────────────────────────────────────────────
    deltas = sorted(((a, tgt[a] - prev.get(a, tgt[a])) for a in tgt),
                    key=lambda kv: -abs(kv[1]))
    if deltas:
        a0, d0 = deltas[0]
        if abs(d0) < 0.03:
            out.append(("전일 대비",
                        f"거의 변화 없습니다 (최대 {short_name(a0)} {d0:+.2f}). "
                        f"position_smooth 0.8 이 오늘 신규 목표의 20%만 반영해 "
                        f"주문이 잘게 나옵니다."))
        else:
            top = [(a, d) for a, d in deltas[:3] if abs(d) >= 0.01]
            out.append(("전일 대비",
                        f"변화가 큰 자산: {_fmt_assets(top)} "
                        f"(스무딩 0.8 적용 후 기준)."))

    # ── ⑤ 북스톱 상태 ──────────────────────────────────────────────────
    st = _stop_line()
    if st:
        out.append(st)

    basis = ("팩터 옆 숫자는 슬리브 z값의 매매 자산 평균에 가중치를 적용한 "
             "컨빅션 기여도입니다."
             if DIRECTIONAL else
             "팩터 옆 숫자는 국가별 슬리브 z값 평균입니다 (이 북은 상대가치 "
             "비중이 커서 전체 평균으로는 설명되지 않습니다).")
    out.append(("읽는 법",
                basis + " 컨빅션은 '방향의 근거'일 뿐 포지션 크기와 비례하지 "
                "않습니다 — 최종 크기는 인버스볼 → 볼타겟 → 스무딩 → 북스톱 → "
                "노출스케일을 거칩니다. 밸류·캐리는 시그널 8종(英日豪 포함) 평균 "
                "대비 상대값이라, 매매하지 않는 4종이 기준선을 만듭니다."))
    return out


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




def attach_pnl_1d(sig_rows, sleeve_snap, rates_per_unit):
    """전일(최신 거래일) 자산별 손익 및 전일 실행 포지션을 행에 부착 (금리 북).

    r['pnl1d_bp'] — bp (기준자본 대비),  r['pnl1d_w'] — 만원 (손익률 × rates_per_unit),
    r['pos_prev'] — 어제 실행된 포지션 (sleeve pos[-2]; T-2 신호 → T-1 실행).
    rates_per_unit 은 델타 열과 같은 '포지션 1.0 = N만원' 계수라 델타·손익의
    기준자본이 구조적으로 어긋날 수 없다 (2026-07-22 정비).
    금리 손익 = 슬리브 엔진 pos[-2]×ret[-1] (비용 차감 전).
    반환: {'rates': {'bp': ..., 'w': ...}} | None.
    """
    sl      = (sleeve_snap or {}).get('pnl_1d') or {}
    sl_prev = (sleeve_snap or {}).get('prev') or {}
    if not sl or not rates_per_unit:
        return None
    for r in sig_rows:
        if r.get('src') == 'sleeve' and r['asset'] in sl:
            r['pnl1d_bp'] = sl[r['asset']] * 1e4
            r['pnl1d_w']  = sl[r['asset']] * rates_per_unit
        if r.get('src') == 'sleeve' and r['asset'] in sl_prev:
            r['pos_prev'] = sl_prev[r['asset']]
    rates_ret = sum(sl.values())
    return {'rates': {'bp': rates_ret * 1e4, 'w': rates_ret * rates_per_unit}}


def _won_fmt(v, width=0):
    """만원 금액 표시. 북 규모에 따라 만원 단위가 0으로 뭉개지므로 자동 소수점.

    금리 북(순델타 한도 기준, 수천만원)과 FX 북(명목 수백억)이 같은 표에 있어
    한쪽만 맞춘 고정 자릿수는 반드시 한쪽을 뭉갠다.
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '-'.rjust(width)
    a = abs(v)
    s = f"{v:+,.2f}" if a < 1 else (f"{v:+,.1f}" if a < 100 else f"{v:+,.0f}")
    return s.rjust(width) if width else s


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


def compute_delta_info(snap, budget_w, gross_budget_w=None, per_unit_override=None):
    """'포지션 1.0 = N만원' 환산 계수 C 와 그에 따른 델타·한도 사용률.

    C 는 델타 열과 손익 열이 공유하는 단 하나의 기준자본이다 (attach_pnl_1d 참조).

    per_unit_override 가 있으면 그 값을 그대로 쓴다. 없으면 기존처럼 한도에서
    역산: C = min(순한도/과거최대|순포지션|, 그로스한도/과거최대 그로스) —
    역사상 최악의 날에도 두 한도를 모두 만족하는 가장 큰 계수.

    ⚠ 자동 역산은 '과거 최대'에 의존하므로 엔진 설정이 바뀌어 포지션 히스토리가
    달라지면 C 가, 따라서 표시되는 모든 델타·손익 금액이 함께 재조정된다
    (2026-07-22 리버전 OFF + 호주 원복만으로 713만원 → 1,072만원). 운용 중
    금액을 고정하려면 --per-unit 으로 못박을 것. 'auto' 플래그로 어느 쪽인지
    표시한다.
    """
    net_hist = snap.get('net_hist')
    if net_hist is None or len(net_hist) == 0:
        return None
    max_net = float(net_hist.abs().max())
    if max_net <= 0:
        return None
    gross_hist = snap.get('gross_hist')
    max_gross = float(gross_hist.max()) if gross_hist is not None and len(gross_hist) else 0.0
    if per_unit_override:
        c, binding, auto = float(per_unit_override), 'fixed', False
    else:
        c, binding, auto = budget_w / max_net, 'net', True
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
            'binding': binding, 'auto': auto}


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




def build_ytd_perf_html(start='2026-01-01'):
    """YTD 성과 plotly div — 금리 북 (sleeve_backtest_log.csv 'rates', net of costs,
    배정자본 대비 %). FX 팩토리 폐기(2026-09-11)로 단일 북이라 볼 정규화 없이
    실제 로그 수익률 그대로 누적한다. 로그 없거나 plotly 미설치면 None."""
    try:
        import numpy as np
        import plotly.graph_objects as go
        slv = pd.read_csv(ROOT / 'sleeve_backtest_log.csv', index_col=0, parse_dates=True)
        s = (slv['rates'] * 100.0).rename('RATES')
        ytd = s[s.index >= pd.to_datetime(start)]
        if len(ytd) < 5:
            return None
        sr = ytd.mean() / ytd.std() * np.sqrt(252) if ytd.std() > 0 else 0.0
        cum = ytd.cumsum().round(3)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ytd.index, y=cum, name=f"RATES (SR {sr:.2f})",
                                 line=dict(width=2.4, color='#4c8ef7')))
        fig.add_annotation(x=ytd.index[-1], y=float(cum.iloc[-1]),
                           text=f"<b>{cum.iloc[-1]:+.2f}%</b>", showarrow=False,
                           xanchor='left', xshift=8, font=dict(color='#4c8ef7', size=13))
        fig.update_layout(
            template='plotly_dark', height=380,
            margin=dict(l=40, r=70, t=48, b=30),
            title=f"YTD 성과 — 금리 북 ({start} ~ · 백테스트 로그 net · 배정자본 대비 %)",
            yaxis_title="누적 PnL (%, 배정 자본 기준)",
            paper_bgcolor='#13161f', plot_bgcolor='#13161f',
            legend=dict(orientation='h', y=1.0, x=0))
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    except Exception as e:
        print(f"⚠️ YTD 성과 차트 생성 실패 (sleeve_backtest_log.csv 필요): {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 콘솔 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_signal_table(sig_rows, signal_date, delta_info=None, pnl_totals=None,
                       sleeve_snap=None):
    print(f"\n▌ 오늘의 트레이딩 시그널  (기준일 {signal_date} · 금리 북 = 슬리브 엔진 — "
          f"main.py --mode signals 동일 · FX 는 매매하지 않음)")
    print(f"  {'자산':<26} {'기초지표':>14} {'가격':>9} {'방향':<8} {'확신도':>6} "
          f"{'포지션':>7} {'델타증감만원':>12} {'델타만원':>8} {'손익bp':>7} {'손익만원':>8} {'출처':>6}")
    last_klass = None
    for r in sig_rows:
        if r['klass'] != last_klass:
            print(f"  ── {CLASS_LABEL.get(r['klass'], r['klass'])} ──")
            last_klass = r['klass']
        arrow = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '· 중립'}[r['dir']]
        ddw  = f"{r['ddelta_w']:>+12.0f}" if 'ddelta_w' in r else f"{'-':>12}"
        dw   = f"{r['delta_w']:>+8.0f}"   if 'delta_w'  in r else f"{'-':>8}"
        pbp  = f"{r['pnl1d_bp']:>+7.1f}"  if 'pnl1d_bp' in r else f"{'-':>7}"
        pw   = _won_fmt(r['pnl1d_w'], 8) if 'pnl1d_w'  in r else f"{'-':>8}"
        px   = _px_fmt(r.get('price'))
        und  = r.get('underlying', '-')
        print(f"  {asset_label(r['asset']):<26} {und:>14} {px:>9} {arrow:<8} {r['conf']:>6.2f} "
              f"{r['pos']:>+7.2f} {ddw} {dw} {pbp} {pw} {'SLV':>6}")
    # 금리 북 시그널 자동 해석 (HTML 과 동일 내용 — sleeve_narrative 단일 소스)
    narr = sleeve_narrative(sleeve_snap, per_unit=(delta_info or {}).get('per_unit'))
    if narr:
        print("\n  🧭 금리 북 시그널 해석  (오늘 슬리브 z값에서 자동 생성)")
        for headline, body in narr:
            print(textwrap.fill(f"{headline} — {body}", width=100,
                                initial_indent='     ', subsequent_indent='       '))
    if pnl_totals and pnl_totals.get('rates'):
        t = pnl_totals['rates']
        print(f"\n  📈 전일 손익 합계: 금리 {t['bp']:+.1f}bp ({_won_fmt(t['w'])}만원)")
    if delta_info:
        d = delta_info
        g_txt = (f" · 그로스 {d['gross_w']:,.0f}만원 / 한도 {d['gross_budget']:,.0f}만원 "
                 f"(사용률 {d['gross_usage']:.0%})" if d.get('gross_budget')
                 else f" · 그로스 {d['gross_w']:,.0f}만원")
        print(f"\n  💰 금리 북 순델타 {d['net_w']:+,.0f}만원 / 한도 ±{d['budget']:,.0f}만원 "
              f"(사용률 {d['usage']:.0%}){g_txt}")
        if d['binding'] == 'fixed':
            basis = "--per-unit 로 고정"
        else:
            basis = (f"과거 최대 |순포지션| {d['max_net_units']:.2f} / 그로스 "
                     f"{d['max_gross_units']:.2f} 중 "
                     f"{'그로스' if d['binding'] == 'gross' else '순델타'} 한도에서 자동 역산 "
                     f"⚠엔진 설정이 바뀌면 이 계수와 위 금액이 모두 재조정됨")
        print(f"     환산: 포지션 1.0 = {d['per_unit']:,.0f}만원 — {basis}")
        print(f"     (델타·손익 모두 이 계수 기준 — 같은 기준자본)")


def print_fx_monitor(fx_snap):
    """FX 팩터 모니터링 — 슬리브 엔진 FX 북의 팩터 z 와 가상 포지션. 매매 안 함."""
    if not fx_snap or not fx_snap.get('target'):
        return
    d = fx_snap['date'].date() if hasattr(fx_snap['date'], 'date') else fx_snap['date']
    print(f"\n▌ FX 팩터 모니터링  (기준일 {d} · 매매 안 함 — 2026-09-11 FX 전략 공장 폐기, "
          f"슬리브 엔진 FX 팩터는 관찰 전용)")
    assets = sorted(fx_snap['target'])
    for name in ('trend', 'value', 'carry'):
        s = fx_snap['sleeves'].get(name)
        if not s:
            continue
        title, desc = FX_SLEEVE_INFO.get(name, (name, ''))
        print(f"  ── {title:<8} (가중 {s['weight']:.1f})  {desc}")
        line = []
        for a in assets:
            v = s['signals'].get(a)
            if v is None:
                continue
            arrow = {'LONG': '▲', 'SHORT': '▼', '-': '·'}[_sleeve_dir(v, 0.1)]
            line.append(f"{short_name(a)} {arrow}{v:+.2f}")
        print("       " + "  ".join(f"{x:<14}" for x in line))
    line = []
    for a in assets:
        p = fx_snap['target'][a]
        arrow = {'LONG': '▲', 'SHORT': '▼', '-': '·'}[_sleeve_dir(p)]
        line.append(f"{short_name(a)} {arrow}{p:+.2f}")
    print(f"  ── {'가상 포지션':<8} (엔진이 매매한다면의 값 — 주문 아님)")
    print("       " + "  ".join(f"{x:<14}" for x in line))


# ─────────────────────────────────────────────────────────────────────────────
# HTML 출력
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




def write_html(signal_date, out_path, sig_rows, sleeve_snap, fx_snap=None,
               delta_info=None, perf_html=None, pnl_totals=None):
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
    .st_on{background:#10381f;color:#34d399;} .st_off{background:#1c1f27;color:#7a828e;} .st_sim{background:#3a2e12;color:#fbbf24;}
    .long{color:#34d399;} .short{color:#f87171;} .flat{color:#555c6b;}
    .type{color:#c4b5fd;} .sh{color:#e6e6e6;font-variant-numeric:tabular-nums;font-weight:500;}
    .desc{color:#cbd2da;} .params{color:#7a828e;font-size:11.5px;}
    hr{border:0;border-top:1px solid #2a2f3a;margin:24px 0;}
    table.strats td:not(.desc), table.strats th{white-space:nowrap;}
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>금리 북 대시보드</title><style>{css}</style></head><body>"]
    parts.append("<h1>금리 북 대시보드 (슬리브 엔진)</h1>")

    # 1. 퀀트 전략 리스트 — 기존 운용 전략(고정) + 폐기된 FX 공장(기록) + 슬리브 금리 북
    parts.append("<h2>[퀀트 전략 리스트]</h2>")
    parts.append("<table class='strats' style='max-width:1200px;margin-bottom:20px;border:1px solid #1e2330;border-radius:6px;overflow:hidden;'>")
    parts.append("<tr style='background:#171b24;'><th>No.</th><th>이름</th><th>자산</th><th>설명</th><th>오버나잇</th><th>상태</th></tr>")

    def _row(idx, name, asset, desc, overnight, status):
        s_class = "st_on" if status == "ON" else ("st_off" if status == "OFF" else "st_sim")
        return (f"<tr><td style='text-align:center;color:#9aa0aa;'>{idx}</td>"
                f"<td style='font-weight:bold;color:#8ab4f8;'>{html.escape(name)}</td>"
                f"<td>{html.escape(asset)}</td><td class='desc'>{html.escape(desc)}</td>"
                f"<td style='text-align:center;'>{html.escape(overnight)}</td>"
                f"<td><span class='pill {s_class}'>{html.escape(status)}</span></td></tr>")

    idx = 1
    parts.append("<tr style='background:#1b1e27;'><td colspan='6' style='font-weight:bold;color:#fbbf24;font-size:12.5px;'>기존 운용 전략</td></tr>")
    for s in STATIC_STRATEGIES:
        parts.append(_row(idx, s['name'], s['asset'], s['desc'], s['overnight'], s['status']))
        idx += 1
    parts.append("<tr style='background:#1b1e27;'><td colspan='6' style='font-weight:bold;color:#7a828e;font-size:12.5px;'>전략 공장 — FX 북 (2026-09-11 폐기)</td></tr>")
    f = RETIRED_FACTORY
    parts.append(_row(idx, f['name'], f['asset'], f['desc'], f['overnight'], f['status']))
    idx += 1
    active_sleeves = set((sleeve_snap or {}).get('sleeves', {}).keys())
    parts.append("<tr style='background:#1b1e27;'><td colspan='6' style='font-weight:bold;color:#c4b5fd;font-size:12.5px;'>슬리브 엔진 — 금리 북 (2016+ 검증 SR 1.16 · 2026-08-14 재검증 · 시그널 8종 / 매매 韓美 4종)</td></tr>")
    for key, name, desc in SLEEVE_STRATS:
        parts.append(_row(idx, name, '글로벌 금리', desc, 'O', 'ON' if key in active_sleeves else 'OFF'))
        idx += 1
    parts.append("</table>")

    # 2. 오늘의 트레이딩 시그널 (자산별 시그널 — 주문 후보)
    if sig_rows:
        parts.append("<h2>[자산별 트레이딩 시그널]</h2>")
        _so = [asset_label(a) for a in ((sleeve_snap or {}).get('signal_only') or [])]
        if _so:
            parts.append(
                "<div class='meta' style='border-left:3px solid #c4b5fd;padding-left:10px;'>"
                "🎯 이 표는 <b>주문 후보</b>입니다 — 금리는 한국·미국 국채선물만 "
                "올라옵니다. "
                f"<b>{html.escape(' · '.join(_so))}</b> 는 집행 시간 제약으로 매매하지 "
                "않으므로 표에서 제외했습니다. 다만 이들은 횡단면 z·demean 기준선을 "
                "만들어 위 한국·미국 포지션을 실제로 좌우하므로, 팩터 값은 "
                "'금리 팩터 시그널 시계열'에 그대로 남겨 뒀습니다. FX 는 2026-09-11 "
                "전략 공장 폐기 이후 매매하지 않습니다 (아래 모니터링 섹션만).</div>")
        parts.append("<div class='meta'>포지션 = 기준자본 대비 <b>명목 배수</b>. "
                     "인버스볼 사이징이 반영돼 저변동 자산이 큰 숫자를 받음 — "
                     "명목이 커도 리스크 기여는 변동성에 비례. "
                     "금리 북 전체는 기준자본의 연 7.1% 변동성으로 타게팅(rates_exposure_scale 0.5 적용 후). "
                     "120일 시계열 = 슬리브 엔진 실시간 포지션.</div>")
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
                + (" (--per-unit 고정)" if d['binding'] == 'fixed' else
                   f"(과거 최대 |순| {d['max_net_units']:.2f} / 그로스 "
                   f"{d['max_gross_units']:.2f} 중 "
                   f"{'그로스' if d['binding'] == 'gross' else '순델타'} 한도에서 자동 역산 — "
                   f"엔진 설정이 바뀌면 이 계수와 금액이 함께 재조정됨)")
                + " · 델타와 손익이 같은 기준자본을 씁니다.</div>")
        if pnl_totals and pnl_totals.get('rates'):
            t = pnl_totals['rates']
            cls = 'long' if t['bp'] > 0 else ('short' if t['bp'] < 0 else 'flat')
            parts.append(
                f"<div class='meta'>📈 전일 손익: 금리 <b class='{cls}' style='font-size:15px;font-weight:700;'>"
                f"{t['bp']:+.1f}bp ({_won_fmt(t['w'])}만원)</b> — 슬리브 분해(비용 차감 전)</div>")
        sig_hist = (sleeve_snap or {}).get('history')
        parts.append("<table style='min-width:1560px;border:1px solid #1e2330;border-radius:6px;overflow:hidden;'>")
        parts.append("<tr style='background:#171b24;white-space:nowrap;'>"
                     "<th>자산</th><th>기초 금리/환율</th><th>방향</th>"
                     "<th>확신도</th><th>전일 포지션</th><th>오늘 포지션</th>"
                     "<th>전일 델타 증감(만원)</th><th>델타(만원)</th>"
                     "<th>전일 손익(bp)</th><th>전일 손익(만원)</th>"
                     "<th>출처</th><th>최근 120일간의 포지션 변동</th></tr>")
        last_k = None
        for r in sig_rows:
            if r['klass'] != last_k:
                parts.append(f"<tr><td colspan='12' class='klass' style='background:#13161f;padding:6px 12px;font-size:13px;'>"
                             f"{CLASS_LABEL.get(r['klass'], r['klass'])}</td></tr>")
                last_k = r['klass']
            dcls = {'LONG': 'long', 'SHORT': 'short', '-': 'flat'}[r['dir']]
            dtxt = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '· 중립'}[r['dir']]
            if 'ddelta_w' in r:
                dp_cls = ('long' if r['ddelta_w'] > 0.5 else
                          ('short' if r['ddelta_w'] < -0.5 else 'flat'))
                dpos_td = (f"<td class='{dp_cls}' style='font-variant-numeric:tabular-nums;'>"
                           f"{r['ddelta_w']:+,.0f}</td>")
            else:
                dpos_td = "<td class='sh'>–</td>"
            dw = f"{r['delta_w']:+,.0f}" if 'delta_w' in r else '–'
            if sig_hist is not None and r['asset'] in sig_hist.columns:
                spark = _spark_svg(sig_hist[r['asset']].tolist())
            else:
                spark = '–'
            px_span = (f" <span class='params'>{_px_fmt(r['price'])}</span>"
                       if r.get('price') is not None else "")
            und = html.escape(r.get('underlying', '–'))
            if 'pos_prev' in r:
                pp = r['pos_prev']
                pp_cls = ('long' if pp > 0.02 else ('short' if pp < -0.02 else 'flat'))
                pp_txt = {'long': '▲', 'short': '▼', 'flat': '·'}[pp_cls]
                pos_prev_td = (f"<td class='{pp_cls}' style='font-variant-numeric:tabular-nums;'>"
                               f"{pp_txt} {pp:+.2f}</td>")
            else:
                pos_prev_td = "<td class='sh'>–</td>"
            if 'pnl1d_bp' in r:
                p_cls = ('long' if r['pnl1d_bp'] > 0.05 else
                         ('short' if r['pnl1d_bp'] < -0.05 else 'flat'))
                pnl_bp_td = (f"<td class='{p_cls}' style='font-variant-numeric:tabular-nums;'>"
                             f"{r['pnl1d_bp']:+.1f}</td>")
                pnl_w_td  = (f"<td class='{p_cls}' style='font-variant-numeric:tabular-nums;'>"
                             f"{_won_fmt(r['pnl1d_w'])}</td>")
            else:
                pnl_bp_td = "<td class='sh'>–</td>"
                pnl_w_td  = "<td class='sh'>–</td>"
            parts.append(f"<tr><td>{html.escape(asset_label(r['asset']))}{px_span}</td>"
                         f"<td class='sh'>{und}</td>"
                         f"<td class='{dcls}'>{dtxt}</td>"
                         f"<td class='sh'>{r['conf']:.2f}</td>"
                         f"{pos_prev_td}"
                         f"<td class='{'long' if r['pos'] > 0.02 else ('short' if r['pos'] < -0.02 else 'flat')}' style='font-variant-numeric:tabular-nums;'>{'▲' if r['pos'] > 0.02 else ('▼' if r['pos'] < -0.02 else '·')} {r['pos']:+.2f}</td>"
                         f"{dpos_td}"
                         f"<td class='sh'>{dw}</td>"
                         f"{pnl_bp_td}"
                         f"{pnl_w_td}"
                         f"<td class='sh'>슬리브</td>"
                         f"<td>{spark}</td></tr>")
        parts.append("</table>")

        # 2-1. 금리 북 시그널 자동 해석 (표 바로 아래) + 기계 판독용 사본
        narr = sleeve_narrative(sleeve_snap,
                                per_unit=(delta_info or {}).get('per_unit'))
        if narr:
            parts.append(
                "<div style='margin:14px 0 8px;padding:14px 16px;background:#12151d;"
                "border:1px solid #232838;border-left:3px solid #8ab4f8;"
                "border-radius:6px;max-width:1560px;'>"
                "<div style='font-weight:bold;color:#8ab4f8;font-size:13.5px;"
                "margin-bottom:10px;'>🧭 금리 북 시그널 해석 "
                "<span style='color:#6b7280;font-weight:normal;font-size:12px;'>"
                "— 오늘 슬리브 z값에서 자동 생성 (수기 코멘트 아님)</span></div>")
            for headline, body in narr:
                muted = headline in ('읽는 법',)
                warn = headline.startswith(('⚠', '⛔'))
                hc = '#f0a868' if warn else ('#6b7280' if muted else '#c4b5fd')
                bc = '#9aa0aa' if muted else '#d5d8de'
                parts.append(
                    f"<div style='margin:6px 0;font-size:{12 if muted else 13}px;"
                    f"line-height:1.65;'>"
                    f"<b style='color:{hc};'>{html.escape(headline)}</b> "
                    f"<span style='color:{bc};'>{html.escape(body)}</span></div>")
            parts.append("</div>")
            # total_dashboard(Streamlit)가 표 아래에 같은 해석을 렌더할 때 읽는다.
            # 표와 같은 파일에서 나오므로 desync 불가; 주석+base64 라 화면·read_html 무영향.
            payload = base64.b64encode(
                json.dumps(narr, ensure_ascii=False).encode('utf-8')).decode()
            parts.append(f"<!--SLEEVE_NARRATIVE_B64:{payload}-->")

    # 2a. YTD 성과 (금리 북)
    if perf_html:
        parts.append("<h2>[YTD 성과 — 금리 북]</h2>")
        parts.append(f"<div style='max-width:1060px;'>{perf_html}</div>")

    # 2b. 슬리브 엔진 (금리 북): 슬리브별 → 자산별
    if sleeve_snap:
        d = sleeve_snap['date'].date() if hasattr(sleeve_snap['date'], 'date') else sleeve_snap['date']
        s_assets = sorted(sleeve_snap['target'])
        parts.append(f"<h2>[슬리브 엔진 — 금리 북]</h2>"
                     f"<div class='meta'>기준일 {d} · 연속 시그널 → 인버스볼 × 포트 볼타게팅 × 스무딩 × 북스톱 · "
                     f"main.py --mode signals 와 동일</div>")
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

    # 3. FX 팩터 모니터링 — 매매 안 함 ('방향'+'오늘 포지션' 조합을 쓰지 않는다: 시그널 표 파서 보호)
    if fx_snap and fx_snap.get('target'):
        d = fx_snap['date'].date() if hasattr(fx_snap['date'], 'date') else fx_snap['date']
        parts.append("<h2>[FX 팩터 모니터링 — 매매 안 함]</h2>"
                     f"<div class='meta'>기준일 {d} · 2026-09-11 FX 전략 공장 폐기 이후 FX 는 "
                     "매매하지 않는다. 아래는 슬리브 엔진 FX 북(추세/밸류/캐리, 2016+ SR 0.33 · "
                     "2022+ 0.10)이 <b>매매한다면</b>의 팩터 z 와 가상 포지션 — 관찰 전용.</div>")
        fx_assets = sorted(fx_snap['target'])
        cols = [n for n in ('trend', 'value', 'carry') if n in fx_snap['sleeves']]
        parts.append("<table style='max-width:900px;'><tr style='background:#171b24;'><th>자산</th><th>통화쌍</th>"
                     + "".join(f"<th>{html.escape(FX_SLEEVE_INFO.get(n, (n, ''))[0])} z</th>" for n in cols)
                     + "<th>가상 포지션</th></tr>")
        for a in fx_assets:
            p = fx_snap['target'][a]
            pair = FX_QUOTE.get(a, ('–', None))[0]
            cells = []
            for n in cols:
                v = fx_snap['sleeves'][n]['signals'].get(a)
                if v is None:
                    cells.append("<td class='sh'>–</td>")
                    continue
                dr = _sleeve_dir(v, 0.1)
                dcls = {'LONG': 'long', 'SHORT': 'short', '-': 'flat'}[dr]
                cells.append(f"<td class='{dcls}' style='font-variant-numeric:tabular-nums;'>{v:+.2f}</td>")
            dr = _sleeve_dir(p)
            dcls = {'LONG': 'long', 'SHORT': 'short', '-': 'flat'}[dr]
            dtxt = {'LONG': '▲ 롱', 'SHORT': '▼ 숏', '-': '· 중립'}[dr]
            parts.append(f"<tr><td>{html.escape(asset_label(a))}</td><td class='sh'>{html.escape(pair)}</td>"
                         + "".join(cells) + f"<td class='{dcls}'>{dtxt} {p:+.2f}</td></tr>")
        parts.append("</table>")

    parts.append("<hr>")
    parts.append(f"<div class='meta'>기준일 {signal_date} · 금리 북 = SleeveEngine (시그널 8종 / 매매 韓美 4종) · "
                 f"FX 전략 공장 2026-09-11 폐기 (git tag pre-fx-factory-retire)</div>")
    parts.append("</body></html>")
    out_path.write_text("".join(parts), encoding='utf-8')
    print(f"\n📊 HTML 대시보드 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def run(asset=None, per_unit=1252.0, delta_budget=5000.0, gross_budget=8000.0,
        perf_start='2026-01-01', html_out=False) -> int:
    """콘솔 시그널 (+ HTML). main.py --mode signals 가 이 함수를 그대로 호출한다."""
    print("📊 가격 데이터 로딩...")
    loader = DataLoader()
    sleeve_snap, fx_snap, prices = build_sleeve_snapshots(loader)
    if sleeve_snap is None:
        print("⚠️ 슬리브 엔진 스냅샷을 만들지 못했습니다 — 종료.")
        return 1

    def _filter(snap):
        if not snap:
            return None
        t = {a: p for a, p in snap['target'].items() if asset.lower() in a.lower()}
        if not t:
            return None
        return {**snap, 'target': t,
                'sleeves': {n: {**s, 'signals': {a: v for a, v in s['signals'].items() if a in t}}
                            for n, s in snap['sleeves'].items()}}

    if asset:
        sleeve_snap, fx_snap = _filter(sleeve_snap), _filter(fx_snap)
        if sleeve_snap is None and fx_snap is None:
            print("⚠️ 해당 자산이 없습니다.")
            return 1

    signal_date = (str(sleeve_snap['date'].date()) if sleeve_snap
                   else str(fx_snap['date'].date()))
    sig_rows, delta_info, pnl_totals = [], None, None
    if sleeve_snap:
        delta_info = compute_delta_info(sleeve_snap, delta_budget,
                                        gross_budget_w=gross_budget,
                                        per_unit_override=per_unit)
        pu = delta_info['per_unit'] if delta_info else None
        sig_rows = sleeve_signal_rows(sleeve_snap, pu)
        last_px = prices.iloc[-1]
        for r in sig_rows:
            if r['asset'] in last_px.index:
                r['price'] = float(last_px[r['asset']])
        attach_underlying(sig_rows, prices)
        pnl_totals = attach_pnl_1d(sig_rows, sleeve_snap, pu)
        print_signal_table(sig_rows, signal_date, delta_info=delta_info,
                           pnl_totals=pnl_totals, sleeve_snap=sleeve_snap)
        print_sleeve_console(sleeve_snap)
    print_fx_monitor(fx_snap)

    if html_out:
        # 산출물은 reports/dashboards/ 아래 (cwd 무관 절대경로). total_dashboard 의
        # _find_dashboard_html 이 이 폴더를 본다. 최신 1개만 유지.
        dash_dir = ROOT / 'reports' / 'dashboards'
        dash_dir.mkdir(parents=True, exist_ok=True)
        out = dash_dir / f"strategy_dashboard_{date.today().isoformat()}.html"
        write_html(signal_date, out, sig_rows, sleeve_snap, fx_snap=fx_snap,
                   delta_info=delta_info, perf_html=build_ytd_perf_html(start=perf_start),
                   pnl_totals=pnl_totals)
        for old in dash_dir.glob('strategy_dashboard_*.html'):
            if old != out:
                old.unlink()
    return 0


def main():
    ap = argparse.ArgumentParser(description="금리 북 대시보드 (슬리브 엔진)")
    ap.add_argument('--asset', default=None, help="특정 자산만 필터 (티커 일부, 예: TU1)")
    ap.add_argument('--per-unit', type=float, default=1252.0,
                    help="금리 '포지션 1.0 = N만원' 환산 계수 (기본 1252, 2026-07-22 "
                         "고정). 델타·손익 양쪽에 같은 계수. 0 = 한도에서 자동 역산 "
                         "(히스토리 의존 → 표시 금액이 흔들림)")
    ap.add_argument('--delta-budget', type=float, default=5000.0,
                    help="금리 북 순델타 한도 (만원, 기본 5000)")
    ap.add_argument('--gross-budget', type=float, default=8000.0,
                    help="금리 북 그로스 한도 (만원, 기본 8000)")
    ap.add_argument('--perf-start', default='2026-01-01',
                    help="HTML YTD 성과 차트 시작일 (기본 2026-01-01)")
    ap.add_argument('--html', action='store_true', help="HTML 대시보드도 저장")
    args = ap.parse_args()
    sys.exit(run(asset=args.asset, per_unit=args.per_unit, delta_budget=args.delta_budget,
                 gross_budget=args.gross_budget, perf_start=args.perf_start,
                 html_out=args.html))


if __name__ == '__main__':
    main()
