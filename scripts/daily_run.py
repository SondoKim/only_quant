"""
일과용 통합 명령어
  python scripts/daily_run.py [options]

단계 0: 캐시 갱신  (어제자 캐시 없으면 구형 삭제 -> Bloomberg 재풀 유도)
단계 1: run_sleeve_backtest.py  (금리 북 로그 -> sleeve_backtest_log.csv,
        금리 팩터 -> sleeve_factor_signals.csv)
단계 2: strategy_dashboard.py --html  (콘솔 시그널 + HTML 대시보드 저장)

2026-09-11 FX 전략 공장 폐기: 옛 1단계(run_backtest.py, FX 북 로그 ~5분)와
3단계(main.py --mode signals)를 제거했다 — 시그널 콘솔 출력은 대시보드 단계가
같은 코드 경로로 찍는다. 전체 실행이 ~15초로 줄었다.

--monitor-only 는 0+1 만 돈다 — 대시보드 '모니터링 실행' 버튼이 '금리 팩터'·
'금리 커브/스프레드'용 캐시(yields/prices parquet + sleeve_factor_signals.csv)만
갱신할 때 쓴다. --skip-fx-bt 는 호환용 무동작 플래그 (옛 호출자가 넘겨도 오류 없음).
"""

import argparse
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))   # run_stamp import 용
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
CACHE_DIR  = ROOT / 'data' / 'cache'
sys.path.insert(0, str(ROOT))
from src.data.loader import default_end_date, SESSION_CUTOFF_HOUR  # noqa: E402

# 일과 시그널 캐시 종류 — 하나라도 어제자가 없으면 전부 재풀 (서로 다른 날짜의
# 가격·일드·매크로·롤 정보가 섞이면 롤 보정·신선도 점검이 어긋난다).
CACHE_KINDS = ('prices', 'yields', 'macro', 'rolls')


# ─────────────────────────────────────────────────────────────────────────────
# 캐시 갱신 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_cache() -> str:
    """
    일과 시그널용 캐시(고정 start_date 2010-01-01 / 2020-01-01 짜리)만 대상으로
    기준일(end_date) 캐시가 없으면 구형 파일을 삭제해 Bloomberg 재풀을 유도한다.

    기준일 = loader.default_end_date() — KST SESSION_CUTOFF_HOUR 시 전이면 그저께
    (미국 세션 미마감 장중가가 하루치 캐시로 고정되는 것을 막는다).
    같은 날 두 번 실행하면 캐시가 이미 있으므로 Bloomberg 재호출 없음.

    반환값: 'refreshed(N)' | 'already_current' | 'skipped(no cache dir)'
    """
    if not CACHE_DIR.exists():
        return 'skipped(no cache dir)'

    end = default_end_date()

    # 일과 시그널 전용 고정 start_date 패턴
    SIGNAL_STARTS = ('2010-01-01', '2020-01-01')

    def kind_of(name: str):
        for k in CACHE_KINDS:
            if any(name.startswith(f'{k}_{s}_') for s in SIGNAL_STARTS):
                return k
        return None

    # 오늘의 캐시(기준일 end_date)가 종류별로 다 있으면 아무것도 안 함
    fresh = {kind_of(f.name) for f in CACHE_DIR.glob('*.parquet')
             if kind_of(f.name) and end in f.name}
    if fresh >= set(CACHE_KINDS):
        return 'already_current'

    # 구형 시그널 캐시(+메타)만 삭제
    removed = 0
    for f in CACHE_DIR.glob('*.parquet'):
        if kind_of(f.name) and end not in f.name:
            f.unlink()
            meta = f.with_suffix('.meta.json')
            if meta.exists():
                meta.unlink()
            removed += 1

    return f'refreshed ({removed} stale signal caches removed, Bloomberg pull next)'


def _freshness() -> dict:
    """방금 갱신된 캐시로 신선도 점검 (loader.freshness_report). 실패해도 일과는 계속."""
    try:
        import logging
        logging.disable(logging.WARNING)
        from src.data.loader import DataLoader
        from src.sleeves.sleeve_engine import SleeveEngine
        from scripts.run_sleeve_backtest import load_sleeve_config, freshness_watch
        ld = DataLoader()
        cfg = load_sleeve_config()
        px = ld.engine_prices(cfg)
        eng = SleeveEngine(px, config=cfg,
                           yields=ld.load_signal_yields(start_date='2010-01-01'),
                           macro=ld.load_signal_macro(start_date='2010-01-01'))
        fr = ld.freshness_report(watch=freshness_watch(eng))
        # 불변식 점검 (scripts/invariants.py) — 신선도 외 FAIL 도 같은 경고 목록에 싣는다
        from scripts.invariants import run_invariants
        pos = eng.finalize_positions(eng.compute_target_positions())
        inv = run_invariants(eng, ld, pos, watch=freshness_watch(eng))
        fr['invariants'] = inv
        fr['issues'] = fr['issues'] + [f'[불변식 {it}] {m}' for it, st, m in inv
                                       if st == 'FAIL' and it != 'fresh']
        return fr
    except Exception as e:
        return {'data_asof': None, 'expected': None, 'issues': [f'신선도/불변식 점검 실패: {e}']}
    finally:
        import logging
        logging.disable(logging.NOTSET)


def _run(cmd: list[str]) -> int:
    # 자식 프로세스가 부모 콘솔 인코딩(Windows cp949)을 물려받으면 이모지 print 에서
    # UnicodeEncodeError 로 죽어 로그가 갱신되지 않는다. UTF-8 I/O 를 강제한다.
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT, env=env)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="일과용 통합 명령어: 캐시 갱신 + 금리 북 백테스트 로그 + 시그널/HTML 대시보드",
    )
    ap.add_argument('--monitor-only', action='store_true',
                    help="캐시 갱신(0)과 금리 북 백테스트(1)만 실행 — 대시보드 "
                         "'모니터링 실행' 버튼용. HTML 대시보드(2)는 건너뛴다")
    ap.add_argument('--skip-fx-bt', action='store_true',
                    help="(호환용, 무동작) FX 전략 공장은 2026-09-11 폐기됨")
    ap.add_argument('--bt-start',      default='2016-01-01',
                    help="백테스트 시작일 (기본 2016-01-01)")
    ap.add_argument('--capital',       type=float, default=500.0,
                    help="금리 북 배정자본 (억원, 기본 500 = PM FACTOR_RATES). DV01·손익 환산 기준")
    ap.add_argument('--per-unit',      type=float, default=None,
                    help="(폐기, 무시 — 2026-10-02 DV01 환산으로 교체)")
    ap.add_argument('--delta-budget',  type=float, default=5000.0,
                    help="순DV01 한도 (만원/bp, 기본 5,000)")
    ap.add_argument('--gross-budget',  type=float, default=8000.0,
                    help="그로스 DV01 한도 (만원/bp, 기본 8,000)")
    ap.add_argument('--perf-start',    default='2026-01-01',
                    help="YTD 성과 시작일 (기본 2026-01-01)")
    ap.add_argument('--no-cache-refresh', action='store_true',
                    help="캐시 갱신 건너뜀 (Bloomberg 없는 환경에서 수동 억제)")
    args = ap.parse_args()

    # 공용 실행 스탬프·뮤텍스 (scripts/run_stamp.py, 2026-09-11): 여러 소비자(total_dashboard·
    # portfolio_management·아침 배치)가 같은 산출물을 쓰므로 이 스크립트 자체가 직렬화하고,
    # 끝나면 스탬프('daily_run', mode full/monitor-only, asof 직전 영업일)를 남긴다.
    import run_stamp as rs
    _mode = 'monitor-only' if args.monitor_only else 'full'
    _started = datetime.now().isoformat(timespec='seconds')
    if not rs.acquire('only_quant_daily', timeout=900, on_wait=lambda: print('  [대기] 다른 daily_run 실행 중 — 뮤텍스 대기')):
        print('  [주의] 뮤텍스 획득 실패(15분) — 락 없이 진행')

    def banner(title: str) -> None:
        print("=" * 64)
        print(title)
        print("=" * 64)

    # ─── 0. 캐시 갱신 ────────────────────────────────────────
    banner("0  캐시 갱신")
    if args.no_cache_refresh:
        print("  --no-cache-refresh: 건너뜀")
        cache_status = 'skipped(manual)'
    else:
        cache_status = _refresh_cache()
        end = default_end_date()
        print(f"  end_date 기준: {end}"
              + (f"  (KST {SESSION_CUTOFF_HOUR}시 전 실행 → 미국 세션 미마감, 그저께까지)"
                 if datetime.now().hour < SESSION_CUTOFF_HOUR else ""))
        print(f"  결과: {cache_status}")
    print()

    # ─── 1. 금리 북 백테스트 (YTD 라인 + 금리 팩터 시계열) ──
    # 항상 실행: 건너뛰면 대시보드 '금리 팩터 모니터링'이 조용히 구버전을 보여준다.
    banner("1  금리 북 백테스트  (run_sleeve_backtest.py)")
    rc_rt = _run([
        'scripts/run_sleeve_backtest.py',
        '--start-date', args.bt_start,
    ])
    print()

    # ─── 1b. 금리 팩터 후보 요약 JSON (factor_summary.py, 2026-09-11) ──
    # 두 모드 모두 실행: 대시보드(total_dashboard 금리 팩터 · PM)는 data/factor_summary.json 만 읽고 표시한다.
    banner("1b 금리 팩터 후보 요약  (factor_summary.py → data/factor_summary.json)")
    rc_fs = _run(['scripts/factor_summary.py']) if rc_rt == 0 else rc_rt
    print()

    # ─── 1c. 데이터 신선도 점검 (2026-10-02) ──────────────────
    banner("1c 데이터 신선도 · 불변식 점검  (scripts/invariants.py)")
    fresh = _freshness()
    print(f"  데이터 기준일 {fresh.get('data_asof')} (기대 세션 {fresh.get('expected')})")
    if fresh.get('invariants'):
        from scripts.invariants import print_invariants
        print_invariants(fresh['invariants'], indent='  ')
    else:
        for m in fresh.get('issues', []):
            print(f"  ⚠⚠ {m}")
    print()

    # ─── 2. 콘솔 시그널 + HTML 대시보드 ──────────────────────
    if args.monitor_only:
        banner("2  시그널/대시보드 — --monitor-only 로 건너뜀")
        rc_db = 0
    else:
        banner("2  시그널 + 대시보드  (strategy_dashboard.py --html)")
        rc_db = _run([
            'scripts/strategy_dashboard.py', '--html',
        ] + [
            '--capital', str(args.capital),
            '--delta-budget', str(args.delta_budget),
            '--gross-budget', str(args.gross_budget),
            '--perf-start',   args.perf_start,
        ])

    # ─── 완료 요약 ───────────────────────────────────────────
    def st(rc: int) -> str:
        return "OK" if rc == 0 else f"ERR({rc})"

    print()
    print("=" * 64)
    print(f"  0 캐시 갱신      : {cache_status}")
    print(f"  1 금리 북 로그   : {st(rc_rt)}")
    print(f"  1b 팩터 요약 JSON: {st(rc_fs)}")
    if args.monitor_only:
        print(f"  2 시그널/대시보드: 건너뜀 (--monitor-only)")
    else:
        print(f"  2 시그널/대시보드: {st(rc_db)}")
    n_iss = len(fresh.get('issues', []))
    print(f"  신선도·불변식     : {'정상' if not n_iss else f'⚠ FAIL {n_iss}건 — 위 1c 확인'}"
          f" (기준일 {fresh.get('data_asof')})")
    print("=" * 64)

    rc = max(rc_rt, rc_fs, rc_db)
    try:
        if rc == 0:
            # asof = 실제 데이터 마지막 날짜 (달력상 직전 영업일이 아니라) — Bloomberg
            # 실패로 옛 캐시를 썼으면 소비자(PM 러너 is_fresh)가 '오래됨'으로 보고 재실행한다.
            rs.write('daily_run', mode=_mode, started=_started, asof=fresh.get('data_asof'),
                     outputs=['sleeve_backtest_log.csv', 'sleeve_factor_signals.csv', 'data/factor_summary.json', 'data/cache/prices_*.parquet',
                              'data/cache/macro_*.parquet'] + ([] if args.monitor_only else ['reports/*.html']),
                     note=f'cache {cache_status}; freshness issues {len(fresh.get("issues", []))}')
            print(f"  스탬프: daily_run mode={_mode} asof={rs.prev_business_day()}")
    finally:
        rs.release('only_quant_daily')
    sys.exit(rc)


if __name__ == '__main__':
    main()
