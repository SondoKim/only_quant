# Global Macro Trading System

LLM 없이 완전 자동화된 글로벌 매크로 금리/FX 트레이딩 시스템

## 🎯 Features

- **자동 전략 생성**: 기술적 지표 + 교차자산 지표의 모든 조합 자동 탐색
- **고속 백테스트**: vectorbt 기반 벡터화 연산
- **전략 공장**: JSON 형식으로 전략 저장/관리
- **성과 기반 필터링**: 
  - Sharpe 3Y > 0.8 → 저장
  - Sharpe 6M > 0.9 → 활성화

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# 전략 탐색 (최초 실행)
python main.py --mode discover

# 일별 업데이트
python main.py --mode update

# 매매 신호 조회
python main.py --mode signals

# 팩토리 요약
python main.py --mode summary
```

## 📊 Supported Assets

**Rates**: US 2Y/10Y, UK 10Y, AU 3Y/10Y, KR 3Y/10Y, DE 2Y/10Y, FR 10Y, IT 10Y, JP 10Y

**FX**: EUR, GBP, JPY, AUD, KRW

## 🏗️ Architecture

```
Data → Indicators → Strategy Generator → Backtester → Factory → Portfolio
```

## 📈 Strategies

| Type | Examples |
|------|----------|
| Momentum | MA Crossover, MACD, Breakout |
| Mean Reversion | Z-Score, RSI Extremes, Bollinger |
| Cross-Asset | Spread Z-Score, Spread RSI |

## License

MIT
