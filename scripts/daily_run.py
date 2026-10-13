"""
일과용 통합 명령어
  python scripts/daily_run.py [options]

단계 0: 캐시 기준일 확인 (loader 가 기준일 캐시가 없으면 Bloomberg 에서 새로 받는다)
단계 1: run_sleeve_backtest.py  (금리 북 로그 → sleeve_backtest_log.csv,
        금리 팩터 → sleeve_factor_signals.csv)
단계 1b: factor_summary.py (→ data/factor_summary.json)
단계 1c: 데이터 신선도 · 불변식 · 절단(미래참조) 점검 (scripts/invariants.py)
단계 2: strategy_dashboard.py --html (콘솔 시그널 + HTML + PM 용 data/rates_signal_table.json)
단계 3: live_record.py (그날 공표한 목표 포지션을 data/live_record.csv 에 추가 — 재작성 안 함)
단계 3b: reconcile_live.py (공표 vs 백테스트 재작성 · 라이브 페이퍼 손익 · PM 환산 · 집행 대사)
마무리: 성공 시 옛 캐시 정리 (기준일 캐시만 남김)

2026-10-13 정비:
  · 옛 캐시는 새 데이터를 받은 '뒤'에만 지운다 — 예전에는 받기 '전'에 지워서 블룸버그
    장애 시 폴백할 캐시가 없어 그날 시그널이 아예 안 나왔다.
  · --monitor-only 도 2·3 단계를 돈다 — HTML/JSON 이 full 실행 때만 갱신돼 PM 이 한 달
    묵은 시그널 표(KE 방향 반대)를 보여주던 문제. (플래그는 스탬프 mode 표시용으로 유지)
  · 배정자본·DV01 한도 기본값은 config/indicators.yaml (capital_억 · dv01_limit).
--skip-fx-bt · --per-unit 은 호환용 무동작 플래그.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'data' / 'cache'
sys.path.insert(0, str(Path(__file__).resolve().parent))   # run_stamp import 용
sys.path.insert(0, str(ROOT))
from src.data.loader import default_end_date, SESSION_CUTOFF_HOUR  # noqa: E402

# 일과 시그널 캐시 종류 (고정 start_date 패턴)
CACHE_KINDS = ('prices', 'pricesraw', 'yields', 'macro', 'rolls')
SIGNAL_STARTS = ('2010-01-01', '2020-01-01')


def _kind_of(name: str):
    for k in CACHE_KINDS:
        if any(name.startswith(f'{k}_{s}_') for s in SIGNAL_STARTS):
            return k
    return None


def _cache_status() -> str:
    """기준일 캐시가 종류별로 이미 있는지 보고만 한다 (삭제하지 않음)."""
    if not CACHE_DIR.exists():
        return 'no cache dir'
    end = default_end_date()
    have = {_kind_of(f.name) for f in CACHE_DIR.glob('*.parquet') if _kind_of(f.name) and end in f.name}
    missing = [k for k in CACHE_KINDS if k not in have]
    return 'already_current' if not missing else f"pull {','.join(missing)} (기준일 {end})"


def _prune_caches() -> int:
    """성공한 실행 뒤에만 호출: 종류별로 기준일 캐시가 있으면 그 외 옛 캐시(+메타)를 지운다.
    기준일 캐시가 없는 종류(블룸버그 실패로 폴백 중)는 아무것도 지우지 않는다."""
    end = default_end_date()
    removed = 0
    for k in CACHE_KINDS:
        files = [f for f in CACHE_DIR.glob('*.parquet') if _kind_of(f.name) == k]
        if not any(end in f.name for f in files):
            continue
        for f in files:
            if end not in f.name:
                f.unlink()
                meta = f.with_suffix('.meta.json')
                if meta.exists():
                    meta.unlink()
                removed += 1
    return removed


def _checks() -> dict:
    """신선도 + 불변식 + 절단 테스트 (실패해도 일과는 계속)."""
    import logging
    try:
        logging.disable(logging.WARNING)
        from src.data.loader import DataLoader
        from src.sleeves.sleeve_engine import SleeveEngine
        from scripts.run_sleeve_backtest import load_sleeve_config, freshness_watch
        from scripts.invariants import run_invariants
        ld = DataLoader()
        cfg = load_sleeve_config()
        px, yl, mc = ld.engine_inputs(cfg)
        eng = SleeveEngine(px, config=cfg, yields=yl, macro=mc)
        fr = ld.freshness_report(watch=freshness_watch(eng))
        pos = eng.finalize_positions(eng.compute_target_positions())
        inv = run_invariants(eng, ld, pos, watch=freshness_watch(eng))
        fr['invariants'] = inv
        fr['issues'] = fr['issues'] + [f'[불변식 {it}] {m}' for it, st, m in inv
                                       if st == 'FAIL' and it != 'fresh']
        return fr
    except Exception as e:
        return {'data_asof': None, 'expected': None, 'issues': [f'신선도/불변식 점검 실패: {e}']}
    finally:
        logging.disable(logging.NOTSET)


def _run(cmd: list) -> int:
    # 자식 프로세스가 부모 콘솔 인코딩(Windows cp949)을 물려받으면 이모지 print 에서
    # UnicodeEncodeError 로 죽어 로그가 갱신되지 않는다. UTF-8 I/O 를 강제한다.
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    return subprocess.run([sys.executable] + cmd, cwd=ROOT, env=env).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="일과용 통합 명령어: 금리 북 로그 + 점검 + 시그널/HTML/JSON + 라이브 기록")
    ap.add_argument('--monitor-only', action='store_true',
                    help="(호환) 스탬프 mode 표시만 다르다 — 2026-10-13 부터 모든 단계를 돈다")
    ap.add_argument('--skip-fx-bt', action='store_true', help="(호환용, 무동작)")
    ap.add_argument('--bt-start', default='2016-01-01', help="백테스트 로그 시작일 (기본 2016-01-01)")
    ap.add_argument('--capital', type=float, default=None, help="배정자본 (억원, 기본 = config capital_억)")
    ap.add_argument('--per-unit', type=float, default=None, help="(폐기, 무시)")
    ap.add_argument('--delta-budget', type=float, default=None, help="순DV01 한도 (만원/bp, 기본 = config)")
    ap.add_argument('--gross-budget', type=float, default=None, help="그로스 DV01 한도 (만원/bp, 기본 = config)")
    ap.add_argument('--perf-start', default='2026-01-01', help="YTD 성과 시작일 (기본 2026-01-01)")
    ap.add_argument('--no-cache-refresh', action='store_true',
                    help="성공 후 옛 캐시 정리를 건너뜀 (Bloomberg 없는 환경)")
    args = ap.parse_args()

    # 공용 실행 스탬프·뮤텍스 (scripts/run_stamp.py): 여러 소비자가 같은 산출물을 쓰므로 직렬화.
    import run_stamp as rs
    _mode = 'monitor-only' if args.monitor_only else 'full'
    _started = datetime.now().isoformat(timespec='seconds')
    if not rs.acquire('only_quant_daily', timeout=900,
                      on_wait=lambda: print('  [대기] 다른 daily_run 실행 중 — 뮤텍스 대기')):
        print('  [주의] 뮤텍스 획득 실패(15분) — 락 없이 진행')

    def banner(title: str) -> None:
        print("=" * 64)
        print(title)
        print("=" * 64)

    try:
        banner("0  캐시 기준일")
        cache_status = _cache_status()
        print(f"  end_date 기준: {default_end_date()}"
              + (f"  (KST {SESSION_CUTOFF_HOUR}시 전 실행 → 미국 세션 미마감, 그저께까지)"
                 if datetime.now().hour < SESSION_CUTOFF_HOUR else ""))
        print(f"  상태: {cache_status}  (없는 종류는 1단계에서 Bloomberg 로 받는다)\n")

        banner("1  금리 북 백테스트  (run_sleeve_backtest.py)")
        rc_rt = _run(['scripts/run_sleeve_backtest.py', '--start-date', args.bt_start])
        print()

        banner("1b 금리 팩터 후보 요약  (factor_summary.py → data/factor_summary.json)")
        rc_fs = _run(['scripts/factor_summary.py']) if rc_rt == 0 else rc_rt
        print()

        banner("1c 데이터 신선도 · 불변식 · 절단 테스트  (scripts/invariants.py)")
        fresh = _checks()
        print(f"  데이터 기준일 {fresh.get('data_asof')} (기대 세션 {fresh.get('expected')})")
        if fresh.get('invariants'):
            from scripts.invariants import print_invariants
            print_invariants(fresh['invariants'], indent='  ')
        else:
            for m in fresh.get('issues', []):
                print(f"  ⚠⚠ {m}")
        print()

        banner("2  시그널 + 대시보드  (strategy_dashboard.py --html → HTML + PM JSON)")
        cmd = ['scripts/strategy_dashboard.py', '--html', '--perf-start', args.perf_start]
        for flag, val in (('--capital', args.capital), ('--delta-budget', args.delta_budget),
                          ('--gross-budget', args.gross_budget)):
            if val is not None:
                cmd += [flag, str(val)]
        rc_db = _run(cmd)
        print()

        banner("3  라이브 기록  (live_record.py → data/live_record.csv, 추가 전용)")
        rc_lr = _run(['scripts/live_record.py']) if rc_rt == 0 else rc_rt
        print()

        banner("3b 실매매 대사  (reconcile_live.py → data/reconcile_report.json, 실패해도 계속)")
        _run(['scripts/reconcile_live.py'])
        print()

        rc = max(rc_rt, rc_fs, rc_db, rc_lr)
        pruned = _prune_caches() if (rc == 0 and not args.no_cache_refresh) else 0

        def st(r: int) -> str:
            return "OK" if r == 0 else f"ERR({r})"

        n_iss = len(fresh.get('issues', []))
        print("=" * 64)
        print(f"  0 캐시           : {cache_status} · 옛 캐시 정리 {pruned}개")
        print(f"  1 금리 북 로그   : {st(rc_rt)}")
        print(f"  1b 팩터 요약 JSON: {st(rc_fs)}")
        print(f"  1c 신선도·불변식 : {'정상' if not n_iss else f'⚠ FAIL {n_iss}건 — 위 1c 확인'}"
              f" (기준일 {fresh.get('data_asof')})")
        print(f"  2 시그널/HTML/JSON: {st(rc_db)}")
        print(f"  3 라이브 기록    : {st(rc_lr)}")
        print("=" * 64)

        if rc == 0:
            # asof = 실제 데이터 마지막 날짜 — Bloomberg 실패로 옛 캐시를 썼으면
            # 소비자(PM 러너 is_fresh)가 '오래됨'으로 보고 재실행한다.
            rs.write('daily_run', mode=_mode, started=_started, asof=fresh.get('data_asof'),
                     outputs=['sleeve_backtest_log.csv', 'sleeve_factor_signals.csv',
                              'data/factor_summary.json', 'data/rates_signal_table.json',
                              'data/live_record.csv', 'reports/dashboards/*.html',
                              'data/cache/prices_*.parquet', 'data/cache/macro_*.parquet'],
                     note=f'cache {cache_status}; checks FAIL {n_iss}')
            print(f"  스탬프: daily_run mode={_mode} asof={fresh.get('data_asof')}")
    finally:
        rs.release('only_quant_daily')
    sys.exit(rc)


if __name__ == '__main__':
    main()
