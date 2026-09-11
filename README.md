# Global Macro Rates Book (SleeveEngine)

블룸버그 데이터로 돌아가는 글로벌 금리 팩터 북. 2026-09-11 FX 전략 공장(조합 탐색·
월별 재선정 방식)을 폐기하고 금리 북 슬리브 엔진만 남겼다 — 과최적화된 전략 조합의
관성을 매매하는 구조라 시장·퀀트 논리가 약하다고 판단. 복구 지점: git tag
`pre-fx-factory-retire`.

## 구조

```
Bloomberg(xbbg) → data/cache parquet (prices 2010+, yields, macro)
      ↓
src/sleeves/sleeve_engine.py   Trend / Value / Carry / Policy 연속 시그널
      → 횡단면 중립화(value·carry) → 인버스볼 → 포트 볼타겟(7.1%) → 스무딩 → 북스톱
      ↓
scripts/run_sleeve_backtest.py  sleeve_backtest_log.csv · sleeve_factor_signals.csv
scripts/strategy_dashboard.py   콘솔 주문 후보 + reports/dashboards/*.html
```

- 시그널 유니버스 8종(美2·英1·日1·豪2·韓2), 매매는 한국·미국 4종 (`signal_only_assets`).
- FX 는 매매하지 않고 슬리브 엔진 FX 팩터를 모니터링 섹션에만 표시한다.
- 설정: `config/indicators.yaml` → `sleeves:`, 티커·일드 맵: `config/assets.yaml`.

## 일과

```bash
python scripts/daily_run.py            # 0 캐시 갱신 → 1 금리 백테스트 → 2 시그널/HTML
python scripts/daily_run.py --monitor-only
python main.py --mode signals          # 콘솔 주문 후보 (대시보드와 같은 코드 경로)
python main.py --mode summary          # 엔진 설정 요약
```

`total_dashboard`(Streamlit, 별도 저장소)가 HTML 대시보드와 두 CSV 를 파싱한다 —
`scripts/strategy_dashboard.py` 상단의 계약을 지킬 것.

## 검증 스크립트 (scripts/)

`audit_lookahead.py`(시차·래치 감사, 시그널 변경 후 필수), `test_switching.py`,
`test_range_regime.py`, `test_sleeve_composition.py`, `test_cleanup_ab.py`,
`test_macro_factors.py`, `test_newinfo_factors.py`, `test_move_overlay.py`,
`test_open_execution.py`, `test_vol_target_split.py`, `test_xs_neutralize.py`,
`test_directionality.py`, `test_signal_only_assets.py`, `test_universe_reduction.py`,
`test_curve_blocks.py`. 판정 규율: 개발 2012-21 / 홀드아웃 2022+, T+2·시간대정직 열,
아슬아슬한 통과 채택 금지.

## 설치

```bash
pip install -r requirements.txt
```
