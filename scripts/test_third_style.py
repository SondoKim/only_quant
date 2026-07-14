"""
제3의 비상관 스타일 후보 테스트 (기존 FX 5종 위)
================================================
정직 규약: pos[t-1] x ret[t], 비용 차감 (FX 1bp, KRW 3bp).

후보:
  S1) FX xs-carry     : 외국 단기금리 - 미국 단기금리, xs-demean z (금리 슬리브의 캐리와 동일 철학)
  S2) FX xs-reversion : 10d 수익률 xs-demean 역방향 z (금리 xs-리버전 서브북과 동일 철학)
  S3) FX xs-momentum  : 126d 수익률 xs-demean z (alpha1 ts-mom과 구분되는 xs 버전)

평가: 단독 SR(반분 포함) + 기존 FX/RATES 북과의 상관 + 3북 균등리스크 결합 SR
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FX = ['AD1 Curncy', 'BP1 Curncy', 'EC1 Curncy', 'JY1 Curncy', 'KRW Curncy']
SHORT_YLD = {                       # 통화별 단기금리 티커
    'AD1 Curncy': 'GTAUD2YR Corp',
    'BP1 Curncy': 'GUKG2 Index',
    'EC1 Curncy': 'GDBR2 Index',
    'JY1 Curncy': 'GJGB2 Index',
    'KRW Curncy': 'GVSK3YR Index',  # 2Y 부재 — 3Y 사용
}
US = 'USGG2YR Index'
COST_BPS = {a: (3.0 if a == 'KRW Curncy' else 1.0) for a in FX}

px = pd.read_parquet(ROOT / 'data/cache/prices_2010-01-01_2026-06-11.parquet')[FX].ffill()
yl = pd.read_parquet(ROOT / 'data/cache/yields_2010-01-01_2026-06-11.parquet').ffill()
ret = px.pct_change()


def zscore(df, win=120):
    return ((df - df.rolling(win).mean()) / df.rolling(win).std()).clip(-2, 2)


def xs_demean(sig):
    return sig.sub(sig.mean(axis=1), axis=0)


def book(signal, smooth=0.8, tv=10.0, start='2016-01-01'):
    """시그널 -> 인버스볼 사이징 -> 볼타게팅 -> 정직 PnL (비용 차감)"""
    iv = 1.0 / (ret.rolling(63).std() * np.sqrt(252))
    pos = (signal * iv).ewm(alpha=1 - smooth).mean()
    # 포트 볼타게팅 (t-1 정보)
    pret = (pos.shift(1) * ret).sum(axis=1)
    pvol = (pret.rolling(63).std() * np.sqrt(252)).shift(1)
    lev = (tv / pvol).clip(upper=5).fillna(0)
    pos = pos.mul(lev, axis=0)
    gross = (pos.shift(1) * ret).sum(axis=1)
    costs = (pos.diff().abs() * pd.Series(COST_BPS) / 1e4).sum(axis=1)
    net = (gross - costs).loc[start:] * 100
    turn = pos.diff().abs().sum(axis=1).loc[start:].mean() * 252
    return net, turn


def f(r):
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0


def report(net, turn, label):
    n = len(net)
    h1, h2 = net.iloc[:n // 2], net.iloc[n // 2:]
    cum = net.cumsum()
    dd = (cum - cum.cummax()).min()
    print(f"  {label:24s} SR {f(net):5.2f}  H1 {f(h1):5.2f}/H2 {f(h2):5.2f}  "
          f"MaxDD {dd:6.1f}%  연회전 {turn:5.0f}x")


# ── 후보 시그널 ──────────────────────────────────────────────
carry_raw = pd.DataFrame({a: yl[SHORT_YLD[a]] - yl[US] for a in FX}).reindex(px.index).ffill()
s_carry = xs_demean(zscore(carry_raw, 252))

s_rev = xs_demean(-zscore(px.pct_change(10), 120))

s_mom = xs_demean(zscore(px.pct_change(126), 252))

print("[제3 스타일 후보 — 단독 성과 (2016+, 정직, 비용 차감)]")
streams = {}
for name, sig, sm in [('S1 FX xs-carry', s_carry, 0.95),
                      ('S2 FX xs-reversion', s_rev, 0.5),
                      ('S3 FX xs-momentum', s_mom, 0.9)]:
    net, turn = book(sig, smooth=sm)
    streams[name] = net
    report(net, turn, name)

# ── 기존 북과의 상관 / 3북 결합 ─────────────────────────────
fac = pd.read_csv(ROOT / 'trading_log.csv', parse_dates=['Date']).set_index('Date')
slv = pd.read_csv(ROOT / 'sleeve_backtest_log.csv', index_col=0, parse_dates=True)
fx_b = fac['total_fx_cumpnl'].diff()
rt_b = slv['rates'] * 100

base = pd.concat({'FXbook': fx_b, 'RTbook': rt_b}, axis=1).dropna()
rv = (base.rolling(63).std() * np.sqrt(252)).shift(1)
basen = (base * (10.0 / rv).clip(upper=10)).dropna()
two = 0.5 * basen['FXbook'] + 0.5 * basen['RTbook']

print(f"\n[기존 2북 결합 기준: SR {f(two):.2f}]")
print(f"{'후보':24s} {'corr(FX북)':>10} {'corr(RT북)':>10} {'3북 SR':>8} {'H1':>6} {'H2':>6}")
for name, net in streams.items():
    al = pd.concat({'n': net, 'f': basen['FXbook'], 'r': basen['RTbook']}, axis=1).dropna()
    nv = (al['n'].rolling(63).std() * np.sqrt(252)).shift(1)
    nn = (al['n'] * (10.0 / nv).clip(upper=10)).fillna(0)
    three = (nn + al['f'] + al['r']) / 3
    n = len(three)
    print(f"{name:24s} {al['n'].corr(al['f']):>10.2f} {al['n'].corr(al['r']):>10.2f} "
          f"{f(three):>8.2f} {f(three.iloc[:n//2]):>6.2f} {f(three.iloc[n//2:]):>6.2f}")
