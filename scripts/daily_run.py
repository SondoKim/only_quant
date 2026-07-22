"""
일과용 통합 명령어
  python scripts/daily_run.py [options]

단계 0: 캐시 갱신  (어제자 캐시 없으면 구형 삭제 -> Bloomberg 재풀 유도)
단계 1: run_backtest.py --skip-discovery  (FX 북 로그 -> trading_log.csv)
단계 2: run_sleeve_backtest.py  (금리 북 로그 -> sleeve_backtest_log.csv)
단계 3: main.py --mode signals  (콘솔 시그널 + Hurst 국면)
단계 4: strategy_dashboard.py --html  (HTML 대시보드 저장)

단계 1~2 는 대시보드 [YTD 성과] plotly 가 읽는 로그를 어제 종가까지 갱신한다.
시간이 없을 땐 --skip-fx-bt 로 1 을 건너뛸 수 있다 (YTD 차트의 FX 라인만 구버전).
단계 2 는 대시보드 '금리 팩터 모니터링' 의 소스(sleeve_factor_signals.csv)라 항상 돈다
(~16초). 예전엔 1~2 를 모두 건너뛰는 --fast 가 있었으나, --skip-fx-bt 대비 16초밖에
아끼지 못하면서 금리 팩터 차트를 조용히 구버전으로 만들어 제거했다.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
CACHE_DIR  = ROOT / 'data' / 'cache'


# ─────────────────────────────────────────────────────────────────────────────
# 캐시 갱신 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_cache() -> str:
    """
    일과 시그널용 캐시(고정 start_date 2010-01-01 / 2020-01-01 짜리)만 대상으로
    어제자 캐시가 없으면 구형 파일을 삭제해 Bloomberg 재풀을 유도한다.

    발굴 엔진의 롤링 윈도우 캐시(prices_2012-..._2015-....parquet 등)는 건드리지 않는다.

    같은 날 두 번 실행하면 캐시가 이미 있으므로 Bloomberg 재호출 없음.

    반환값: 'refreshed(N)' | 'already_current' | 'skipped(no cache dir)'
    """
    if not CACHE_DIR.exists():
        return 'skipped(no cache dir)'

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # 일과 시그널 전용 고정 start_date 패턴
    SIGNAL_STARTS = ('2010-01-01', '2020-01-01')

    def is_signal_cache(name: str) -> bool:
        return any(name.startswith(f'prices_{s}_') or name.startswith(f'yields_{s}_')
                   for s in SIGNAL_STARTS)

    # 오늘의 캐시(어제 end_date)가 이미 있으면 아무것도 안 함
    fresh = [f for f in CACHE_DIR.glob('*.parquet')
             if is_signal_cache(f.name) and yesterday in f.name]
    if len(fresh) >= 2:  # prices + yields 각 1개 이상
        return 'already_current'

    # 구형 시그널 캐시만 삭제
    removed = 0
    for f in CACHE_DIR.glob('*.parquet'):
        if is_signal_cache(f.name) and yesterday not in f.name:
            f.unlink()
            removed += 1

    return f'refreshed ({removed} stale signal caches removed, Bloomberg pull next)'


def _run(cmd: list[str]) -> int:
    # 자식 프로세스가 부모 콘솔 인코딩(Windows cp949)을 물려받으면 이모지 print 에서
    # UnicodeEncodeError 로 죽어 백테스트 로그가 갱신되지 않는다 (YTD 차트가 하루 뒤처짐).
    # UTF-8 I/O 를 강제해 콘솔 인코딩과 무관하게 모든 단계가 끝까지 돌도록 한다.
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    result = subprocess.run([sys.executable] + cmd, cwd=ROOT, env=env)
    return result.returncode


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="일과용 통합 명령어: 캐시 갱신 + 백테스트 로그 + 시그널 + HTML 대시보드",
    )
    ap.add_argument('--skip-fx-bt', action='store_true',
                    help="FX 북 백테스트(1단계)만 건너뜀 — 금리 백테스트(2단계)는 돌려 "
                         "금리 신호 차트는 전일까지 갱신 (YTD의 FX 라인만 구버전)")
    ap.add_argument('--bt-start',      default='2016-01-01',
                    help="백테스트 시작일 (기본 2016-01-01)")
    ap.add_argument('--max-corr',      type=float, default=1.0,
                    help="FX 전략 상관 필터 (기본 1.0 = 꺼짐)")
    ap.add_argument('--per-unit',      type=float, default=1252.0,
                    help="금리 '포지션 1.0 = N만원' 환산 계수 (기본 1252, 2026-07-22 "
                         "고정). 델타·손익 공통 기준자본. 0 = 한도에서 자동 역산"
                         "(히스토리 의존 → 표시 금액이 흔들림). 이전의 "
                         "--capital(500억) 은 손익 열에만 쓰여 델타와 1만배 어긋나 "
                         "있었다 — 제거됨")
    ap.add_argument('--fx-notional',   type=float, default=300.0,
                    help="FX 자산별 명목금액 (만달러, 기본 300만달러)")
    ap.add_argument('--fx-usdkrw',     type=float, default=1500.0,
                    help="FX 만원 환산 기준환율 (KRW/USD, 기본 1500)")
    ap.add_argument('--delta-budget',  type=float, default=5000.0,
                    help="순델타 한도 (만원, 기본 5,000)")
    ap.add_argument('--gross-budget',  type=float, default=8000.0,
                    help="그로스 한도 (만원, 기본 8,000)")
    ap.add_argument('--perf-start',    default='2026-01-01',
                    help="YTD 성과 시작일 (기본 2026-01-01)")
    ap.add_argument('--no-cache-refresh', action='store_true',
                    help="캐시 갱신 건너뜀 (Bloomberg 없는 환경에서 수동 억제)")
    args = ap.parse_args()

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
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"  end_date 기준: {yesterday}")
        print(f"  결과: {cache_status}")
    print()

    # ─── 1. FX 북 백테스트 (YTD 차트의 FX 라인) ─────────────
    if args.skip_fx_bt:
        banner("1  FX 북 백테스트 — --skip-fx-bt 로 건너뜀 (YTD 차트 FX 라인은 마지막 실행분)")
        rc_fx = 0
        print()
    else:
        banner("1  FX 북 백테스트  (run_backtest.py --skip-discovery)")
        rc_fx = _run([
            'scripts/run_backtest.py',
            '--start-date', args.bt_start,
            '--skip-discovery',
            '--max-corr', str(args.max_corr),
        ])
        print()

    # ─── 2. 금리 북 백테스트 (YTD RATES 라인 + 금리 신호 차트) ──
    # 항상 실행한다. 실측 ~16초로 저렴한 반면, 건너뛰면 대시보드 '금리 팩터 모니터링'이
    # 조용히 구버전 시그널을 보여준다 → 과거의 --fast 플래그를 제거한 이유.
    banner("2  금리 북 백테스트  (run_sleeve_backtest.py)")
    rc_rt = _run([
        'scripts/run_sleeve_backtest.py',
        '--start-date', args.bt_start,
    ])
    print()

    # ─── 3. 콘솔 시그널 + Hurst 국면 ─────────────────────────
    banner("3  시그널  (main.py --mode signals)")
    rc1 = _run([
        'main.py', '--mode', 'signals',
        '--max-corr', str(args.max_corr),
    ])
    print()

    # ─── 4. HTML 대시보드 저장 ───────────────────────────────
    banner("4  대시보드  (strategy_dashboard.py --html)")
    rc2 = _run([
        'scripts/strategy_dashboard.py', '--html',
        '--max-corr',     str(args.max_corr),
    ] + (['--per-unit', str(args.per_unit)] if args.per_unit else []) + [
        '--fx-notional',  str(args.fx_notional),
        '--fx-usdkrw',    str(args.fx_usdkrw),
        '--delta-budget', str(args.delta_budget),
        '--gross-budget', str(args.gross_budget),
        '--perf-start',   args.perf_start,
    ])

    # ─── 완료 요약 ───────────────────────────────────────────
    def st(rc: int) -> str:
        return "OK" if rc == 0 else f"ERR({rc})"

    print()
    print("=" * 64)
    print(f"  0 캐시 갱신    : {cache_status}")
    if args.skip_fx_bt:
        print(f"  1 FX 북 로그   : 건너뜀 (--skip-fx-bt)")
    else:
        print(f"  1 FX 북 로그   : {st(rc_fx)}")
    print(f"  2 금리 북 로그 : {st(rc_rt)}")
    print(f"  3 시그널       : {st(rc1)}")
    print(f"  4 대시보드     : {st(rc2)}")
    print("=" * 64)

    sys.exit(max(rc_fx, rc_rt, rc1, rc2))


if __name__ == '__main__':
    main()
