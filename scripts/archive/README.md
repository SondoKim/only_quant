# scripts/archive — 종료된 A/B 실험 기록 (2026-10-13 보관)

여기 있는 스크립트는 **이미 결론이 난 실험의 기록**이다. 현재 코드로는 돌지 않거나(삭제된 엔진
기능 사용), 돌더라도 운용과 다른 입력(롤 보정 전 가격, 옛 손익 기준)을 쓴다. 결론과 수치는
메모리(`~/.claude/projects/.../memory/`)와 각 파일 docstring 에 남아 있다.

재현이 필요하면 엔진 정리 전 태그에서 실행한다:

```
git worktree add ../only_quant_pre_cleanup pre-engine-cleanup-2026-10
cd ../only_quant_pre_cleanup && python scripts/test_switching.py
```

새 연구는 이 파일들을 복사하지 말고 운용 경로를 쓸 것:
`DataLoader.engine_inputs(cfg)` (입력) · `SleeveEngine` (포지션) · `src/sleeves/accounting.py`
(손익) · `scripts/invariants.py` (불변식·절단 테스트) — 사본마다 회계가 달라 생긴 2026-10-02
'부풀린 SR' 사고를 반복하지 않기 위해서다.
