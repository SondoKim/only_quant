"""공용 실행 스탬프 + 파일 뮤텍스 (2026-09-11) — 여러 소비자(portfolio_management, total_dashboard,
아침 배치)가 같은 생산자 잡(daily_run.py, 엔진 실행, str_alarm, fs_bridge …)을 공유할 때
'오늘 이미 돌았는가'를 확인하고, 안 돌았으면 하나만 돌리게 직렬화한다.

표준 라이브러리만 쓴다 — 어느 저장소에서든 sys.path 에 이 디렉토리를 넣고 import 한다.
    import run_stamp as rs
    if not rs.is_fresh('daily_run', modes=('full', 'monitor-only')):
        if rs.acquire('only_quant_daily', timeout=900):
            try: ...실행...; rs.write('daily_run', mode='monitor-only')
            finally: rs.release('only_quant_daily')

스탬프 파일: <only_quant>/data/run_stamps/<name>.json
    {name, mode, asof(기준일 YYYY-MM-DD), started, finished, pid, host, outputs, note}
기준일(asof) 규약: 생산자가 '전일 종가'까지 반영했으면 asof = 직전 영업일(월~금, 휴일 미고려).
뮤텍스 파일: <only_quant>/data/run_stamps/mutex_<name>.lock — 내용 = 소유 pid. 죽은 pid 는 자동 정리.
CLI:  python run_stamp.py show            스탬프 목록
      python run_stamp.py fresh <name>    신선하면 0, 아니면 1 (배치 스크립트용)
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

STAMP_DIR = Path(__file__).resolve().parent.parent / 'data' / 'run_stamps'


def prev_business_day(today: datetime | None = None) -> str:
    d = (today or datetime.now()).date()
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime('%Y-%m-%d')


def _path(name: str) -> Path:
    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    return STAMP_DIR / f'{name}.json'


def write(name: str, mode: str = 'full', asof: str | None = None, outputs: list | None = None,
          started: str | None = None, note: str = '') -> dict:
    st = {'name': name, 'mode': mode, 'asof': asof or prev_business_day(),
          'started': started or datetime.now().isoformat(timespec='seconds'),
          'finished': datetime.now().isoformat(timespec='seconds'),
          'pid': os.getpid(), 'host': socket.gethostname(),
          'outputs': [str(o) for o in (outputs or [])], 'note': note}
    _path(name).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
    return st


def read(name: str) -> dict | None:
    p = _path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def is_fresh(name: str, asof: str | None = None, modes: tuple | list | None = None) -> bool:
    """스탬프가 있고 asof(기본 직전 영업일) 이상이며 mode 가 허용 목록에 있으면 True."""
    st = read(name)
    if not st:
        return False
    if st.get('asof', '') < (asof or prev_business_day()):
        return False
    return (modes is None) or (st.get('mode') in modes)


def _pid_alive(pid: int) -> bool:
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == 259                              # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def _mutex_path(name: str) -> Path:
    STAMP_DIR.mkdir(parents=True, exist_ok=True)
    return STAMP_DIR / f'mutex_{name}.lock'


def acquire(name: str, timeout: float = 600.0, on_wait=None) -> bool:
    """파일 락 — 성공 True. timeout 초과 False (호출자가 락 없이 진행할지 결정).
    죽은 소유자의 락은 자동 정리. on_wait 는 대기 시작 시 한 번 호출."""
    path = _mutex_path(name)
    deadline = time.time() + timeout
    waited = False
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = int(path.read_text() or '0')
            except (OSError, ValueError):
                owner = 0
            if owner and not _pid_alive(owner):
                try:
                    if int(path.read_text() or '0') == owner:
                        path.unlink()
                except (OSError, ValueError):
                    pass
                continue
            if time.time() >= deadline:
                return False
            if not waited:
                waited = True
                if on_wait:
                    on_wait()
            time.sleep(3)


def release(name: str) -> None:
    try:
        _mutex_path(name).unlink()
    except OSError:
        pass


def owner(name: str) -> int | None:
    p = _mutex_path(name)
    try:
        return int(p.read_text() or '0') if p.exists() else None
    except (OSError, ValueError):
        return None


def show() -> str:
    rows = []
    for p in sorted(STAMP_DIR.glob('*.json')) if STAMP_DIR.exists() else []:
        st = read(p.stem) or {}
        rows.append(f"{p.stem:14s} asof {st.get('asof', '-')}  mode {st.get('mode', '-'):13s} finished {st.get('finished', '-')}")
    locks = [f"  lock {p.name} (pid {p.read_text() or '?'})" for p in STAMP_DIR.glob('mutex_*.lock')] if STAMP_DIR.exists() else []
    return '\n'.join(rows + locks) or '(스탬프 없음)'


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'fresh':
        sys.exit(0 if is_fresh(sys.argv[2]) else 1)
    print(show())
