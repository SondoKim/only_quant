"""
북 간 배분(FX vs RATES) 동적 조정 A/B 테스트
============================================
정직성 원칙: 가중치는 t-1까지 정보만 사용 (shift 1).

변형:
  A) 50/50 고정 (현행)
  B) 롤링 252d SR 비례 (음수 SR -> 0), 50% 균등 수축
  C) 롤링 126d SR 비례, 50% 수축
  D) 63d PnL 모멘텀 틸트 (최근 좋은 북 +-20%p)
  E) 인샘플 정적 최적 (치팅 기준선 — 상한 참고용)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

fac = pd.read_csv(ROOT / 'trading_log.csv', parse_dates=['Date']).set_index('Date')
slv = pd.read_csv(ROOT / 'sleeve_backtest_log.csv', index_col=0, parse_dates=True)

fx = fac['total_fx_cumpnl'].diff()
rt = slv['rates'] * 100          # 금리 북 일별 수익률 (%)

df = pd.concat({'fx': fx, 'rt': rt}, axis=1).dropna()
print(f"공통 구간: {df.index[0].date()} ~ {df.index[-1].date()}  ({len(df)}d)")

# 롤링 볼 정규화 (t-1까지 정보, 10% 연율 타깃)
TV = 10.0
roll_vol = df.rolling(63).std() * np.sqrt(252)
norm = (df * (TV / roll_vol.shift(1))).dropna()


def perf(ret, label):
    sr = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    cum = ret.cumsum()
    dd = (cum - cum.cummax()).min()
    n = len(ret)
    h1, h2 = ret.iloc[:n // 2], ret.iloc[n // 2:]
    sr1 = h1.mean() / h1.std() * np.sqrt(252) if h1.std() > 0 else 0
    sr2 = h2.mean() / h2.std() * np.sqrt(252) if h2.std() > 0 else 0
    print(f"  {label:36s} SR {sr:5.2f}  MaxDD {dd:6.1f}%  H1 {sr1:5.2f} / H2 {sr2:5.2f}")
    return sr


def run(w_fx, label):
    """w_fx: 시계열 or 스칼라 (이미 t-1 정보 기준이어야 함)"""
    if np.isscalar(w_fx):
        w = pd.Series(w_fx, index=norm.index)
    else:
        w = w_fx.reindex(norm.index).fillna(0.5)
    ret = w * norm['fx'] + (1 - w) * norm['rt']
    return perf(ret, label)


print("\n[북 배분 A/B — 모든 가중치 t-1 정보만 사용]")
run(0.5, "A) 50/50 고정 (현행)")

# B/C) 롤링 SR 비례, 50% 균등수축
for win, tag in [(252, 'B) 252d'), (126, 'C) 126d')]:
    mu = norm.rolling(win).mean()
    sd = norm.rolling(win).std()
    sr_roll = (mu / sd * np.sqrt(252)).clip(lower=0)
    raw = sr_roll['fx'] / (sr_roll['fx'] + sr_roll['rt'])
    raw = raw.fillna(0.5)
    w = (0.5 * raw + 0.5 * 0.5).shift(1)          # 50% 수축 + 1일 지연
    run(w, f"{tag} 롤링SR 비례 (50% 수축)")

# D) 63d PnL 모멘텀 틸트 ±20%p
mom = norm.rolling(63).sum()
tilt = np.where(mom['fx'] > mom['rt'], 0.7, 0.3)
w = pd.Series(tilt, index=norm.index).shift(1)
run(w, "D) 63d PnL 모멘텀 틸트 (70/30)")

# E) 인샘플 정적 최적 (치팅 — 상한 참고)
best_sr, best_w = -9, 0.5
for wf in np.arange(0, 1.01, 0.05):
    ret = wf * norm['fx'] + (1 - wf) * norm['rt']
    sr = ret.mean() / ret.std() * np.sqrt(252)
    if sr > best_sr:
        best_sr, best_w = sr, wf
print(f"\n  E) 인샘플 정적 최적 (치팅): w_fx={best_w:.2f}  SR {best_sr:.2f}")

# 참고: 개별 북
print("\n[개별 북 (볼정규화 후)]")
perf(norm['fx'], "FX 단독")
perf(norm['rt'], "RATES 단독")
