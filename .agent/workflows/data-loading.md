---
description: 블룸버그 xbbg 라이브러리를 이용한 데이터 로드 지침 (새 프로젝트 시작 시 참조)
---

# Bloomberg xbbg 데이터 로드 표준 패턴

새로운 프로젝트나 스크립트에서 **금융 시장 데이터를 로드**할 때, 반드시 아래의 표준 패턴을 따른다.

## 핵심 원칙

1. **라이브러리**: `xbbg` 패키지의 `blp` 모듈을 사용한다.
2. **함수**: `blp.bdh()` (Bloomberg Data History)를 사용하여 historical data를 가져온다.
3. **참조 구현**: `src/data/loader.py`의 `DataLoader` 클래스를 표준 패턴으로 삼는다.

## 표준 데이터 로드 코드

```python
from xbbg import blp

# blp.bdh()로 historical data 로드
df = blp.bdh(
    tickers=['TICKER1 Index', 'TICKER2 Curncy'],  # Bloomberg ticker 형식
    flds=['px_last'],           # 가격 필드 (종가)
    start_date='2020-01-01',    # 시작일 (YYYY-MM-DD)
    end_date='2025-12-31',      # 종료일 (YYYY-MM-DD)
    Per='D',                    # 주기: D(일), W(주), M(월)
    Fill='NA'                   # 빈 값 처리
)

# MultiIndex 컬럼 정리
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

# 일자 인덱스 정리 및 결측치 처리
df.index = pd.to_datetime(df.index)
df = df.ffill().dropna()
```

## xbbg import 안전 패턴

Bloomberg 터미널이 없는 환경에서도 실행 가능하도록 아래 패턴으로 import한다:

```python
try:
    from xbbg import blp
    XBBG_AVAILABLE = True
except ImportError:
    XBBG_AVAILABLE = False
    logger.warning("xbbg not installed. Bloomberg data loading disabled.")
```

## 캐싱 패턴

Bloomberg 호출을 최소화하기 위해 parquet 캐시를 사용한다:

```python
cache_file = cache_dir / f"prices_{start_date}_{end_date}.parquet"

# 캐시 존재 시 로드
if cache_file.exists():
    df = pd.read_parquet(cache_file)

# 신규 로드 후 캐시 저장
df.to_parquet(cache_file)
```

## Bloomberg Ticker 형식 참조

| 자산 유형 | Ticker 형식 예시 | 설명 |
|-----------|------------------|------|
| 금리 (국채) | `USGG2YR Index`, `GT10 Govt` | 수익률 (yield level) |
| FX | `EUR Curncy`, `JPY Curncy` | 통화 |
| 주가지수 | `NQ1 Index`, `SPX Index` | 지수/선물 |
| 채권 스프레드 | `GDBR10 Index` | 스프레드 |

## 체크리스트

새 프로젝트에서 데이터 로더를 만들 때 확인:

- [ ] `xbbg`의 `blp.bdh()` 사용 여부
- [ ] `px_last` 필드 기본 사용
- [ ] MultiIndex 컬럼 정리 로직 포함
- [ ] `ffill().dropna()` 결측치 처리
- [ ] Parquet 캐싱 구현
- [ ] xbbg import 안전 처리 (try/except)
- [ ] `assets.yaml` 기반 ticker 관리 (해당 시)
