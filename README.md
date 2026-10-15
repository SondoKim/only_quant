# Global Macro Rates Book (SleeveEngine)

> **운용 위치 (2026-10-15~)**: FACTOR_RATES 운용 코드는 `D:\김선도\Python\live_strategies` 의 `factor_rates` 로 옮겼다
> (이 레포의 src·scripts 를 계산 그대로 옮김). PM 러너가 그 코드를 고정 커밋 스냅샷에서 돌리고, 블룸버그 패널은
> PM `market_panels.py` 가 `D:\김선도\Python\market_data\panels` 에 받는다. PM·total_dashboard 는 더 이상 이 레포의
> data/cache·sleeve_backtest_log.csv·data/*.json 을 읽지 않는다. 이 레포는 금리 팩터 연구·도구(live_record·reconcile_live·
> PPT 등) 레포로 남는다. 개선 연구는 live_strategies factor_rates 코드를 불러와 시작하고, 채택하면 그쪽에 커밋한 뒤
> PM `pm_config.LIVE_PINS['FACTOR_RATES']` 로 반영한다.

블룸버그 데이터로 돌아가는 글로벌 금리 팩터 북. 2026-09-11 FX 전략 공장(조합 탐색·
월별 재선정 방식)을 폐기하고 금리 북 슬리브 엔진만 남겼다 — 복구 지점: git tag
`pre-fx-factory-retire`. 2026-10-13 엔진에서 기각된 실험 기능을 걷어냈다 — 복구 지점:
git tag `pre-engine-cleanup-2026-10`.

## 구조

```
Bloomberg(xbbg) → data/cache parquet (pricesraw/prices 2010+, yields, macro, rolls)
      ↓  DataLoader.engine_inputs(cfg)  — 엔진 입력 단일 경로 (롤 보정·0가격 제거·제외 자산 처리)
src/sleeves/sleeve_engine.py   Trend / Value / Carry(환헤지 국가간) / Policy 연속 시그널
      → 횡단면 중립화(value·carry) → 인버스볼 → 북별 볼타겟 → 스무딩 → 북스톱 → 노출 스케일 → DV01 한도
src/sleeves/accounting.py      손익 회계 단일 출처 (futures 기준, 비용, 시차 lag_map)
      ↓
scripts/run_sleeve_backtest.py  sleeve_backtest_log.csv · sleeve_factor_signals.csv (PM FACTOR_RATES 원천)
scripts/strategy_dashboard.py   콘솔 주문 후보 + reports/dashboards/*.html + data/rates_signal_table.json (PM)
scripts/live_record.py          그날 공표 목표를 data/live_record.csv 에 추가 (과거 행 불변)
scripts/reconcile_live.py       공표 vs 백테스트 재작성 · 라이브 페이퍼 손익 · PM 환산 · 집행 대사
scripts/invariants.py           불변식 + 절단(미래참조) 테스트 — daily_run 1c · audit ⑥⑦
```

- 시그널 유니버스 8종(美2·英1·日1·豪2·韓2), 매매는 한국·미국 4종 (`signal_only_assets`).
- 배정자본(`capital_억`)·DV01 한도(`dv01_limit`)의 단일 출처 = `config/indicators.yaml` → PM 도 읽는다.
- DV01 β = 선물 5일 수익률 ÷ 테너 금리 5일 bp 변화 회귀(250일). 엔진 한도·대시보드·라이브 기록과
  PM 계약당 DV01(TU·TY·KTB3·KTB10)이 같은 값을 쓴다 — `data/rates_signal_table.json` `durations`.
- 일과 점검(1c) 결과 `data/daily_checks.json` → HTML 배너 · PM JSON `checks` · live_record `checks_fail`.
  `data/live_record.csv` 는 추가 전용 (같은 기준일 재공표는 `rev` 를 올려 새 행).
- FX 는 매매하지 않고 슬리브 엔진 FX 팩터를 모니터링 섹션에만 표시한다.
- 설정: `config/indicators.yaml` → `sleeves:`, 티커·일드 맵: `config/assets.yaml`.

## 일과

```bash
python scripts/daily_run.py            # 1 백테스트 → 1b 팩터요약 → 1c 점검 → 2 시그널/HTML/JSON → 3 라이브기록 → 3b 대사
python main.py --mode signals          # 콘솔 주문 후보 (대시보드와 같은 코드 경로)
python main.py --mode summary          # 엔진 설정 요약
python scripts/audit_lookahead.py      # 시차·래치·비용·불변식·절단 테스트 (시그널·설정 변경 후 필수)
```

`total_dashboard`(Streamlit) 가 HTML 의 전략 리스트를, `portfolio_management` 가 시그널 JSON
(없으면 HTML)과 두 CSV 를 읽는다 — `scripts/strategy_dashboard.py` 상단의 계약을 지킬 것.

## 연구 규율

새 연구는 `scripts/research/` 에, 반드시 운용 경로(`engine_inputs` · `SleeveEngine` ·
`accounting`)를 써서 만든다. 후보와 판정 기준은 실행 전에 docstring 에 고정(사전등록), 개발
2012-21 / 홀드아웃 2022+, T+2·시간대정직 열, 아슬아슬한 통과 채택 금지. 종료된 실험은
`scripts/archive/` (README 참조). 같은 표본에서 수십 개 변형을 시험해 왔으므로 보고 SR 은
낙관적으로 읽을 것.

## 설치

```bash
pip install -r requirements.txt
```
