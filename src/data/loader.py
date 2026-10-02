"""
Bloomberg Data Loader for Global Macro Trading

Loads historical price data from Bloomberg using xbbg library.
Includes caching and offline fallback support.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bloomberg xbbg import
try:
    from xbbg import blp
    XBBG_AVAILABLE = True
except ImportError:
    XBBG_AVAILABLE = False
    logger.warning("xbbg not installed. Bloomberg data loading disabled.")

# 미국 세션 마감(CME 정산 14:00 CT, Globex 16:00 CT) = KST 05:00~07:00. 이 시각
# 이전에 돌리면 '어제' 미국 세션이 아직 진행 중이라 PX_LAST 가 장중가로 들어오고,
# 캐시 이름이 그 날짜로 고정돼 하루 종일 재사용된다 → 컷오프 전에는 그저께까지만.
SESSION_CUTOFF_HOUR = 8


def default_end_date(now: Optional[datetime] = None) -> str:
    """캐시·Bloomberg 조회의 기본 end_date — '완전히 마감된 마지막 세션' 날짜.

    KST SESSION_CUTOFF_HOUR 시 이전이면 그저께, 이후면 어제. 로더와 daily_run 의
    캐시 갱신이 같은 함수를 써야 캐시 이름이 어긋나지 않는다.
    """
    now = now or datetime.now()
    lag = 2 if now.hour < SESSION_CUTOFF_HOUR else 1
    return (now - timedelta(days=lag)).strftime("%Y-%m-%d")


def expected_last_session(end_date: Optional[str] = None) -> pd.Timestamp:
    """end_date 이하의 마지막 평일 (휴장일은 모름 — 신선도 경고 문구에서 감안)."""
    d = pd.Timestamp(end_date or default_end_date())
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _write_meta(cache_file: Path, raw: pd.DataFrame) -> None:
    """ffill 전 원자료의 티커별 마지막 실데이터 날짜를 캐시 옆 JSON 에 남긴다.

    캐시는 ffill 된 상태로 저장되므로(하위 소비자가 마지막 행을 바로 읽음) 캐시만
    봐서는 '휴장/누락으로 전일값이 복사된 칸'을 구분할 수 없다 — 그 정보를 보존.
    """
    try:
        lv = {c: (str(raw[c].last_valid_index().date())
                  if raw[c].last_valid_index() is not None else None)
              for c in raw.columns}
        cache_file.with_suffix('.meta.json').write_text(
            json.dumps({'last_valid': lv,
                        'pulled': datetime.now().isoformat(timespec='seconds')},
                       ensure_ascii=False, indent=1), encoding='utf-8')
    except Exception as e:  # 메타 실패가 데이터 로드를 막아선 안 된다
        logger.warning(f"meta 기록 실패 {cache_file.name}: {e}")


def _read_meta(cache_file: Path) -> Dict[str, Any]:
    p = cache_file.with_suffix('.meta.json')
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


class DataLoader:
    """Load and cache Bloomberg data for global macro trading."""
    
    def __init__(self, config_path: str = None, cache_dir: str = None):
        """
        Initialize DataLoader.
        
        Args:
            config_path: Path to assets.yaml configuration
            cache_dir: Directory for caching data
        """
        self.config_path = config_path or self._get_default_config_path()
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent.parent / 'data' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = self._load_config()
        self.all_tickers = self._extract_all_tickers()
        # 이 로더 인스턴스가 실제로 읽은 캐시 기록 — freshness_report() 가 쓴다.
        # {kind: {'file': Path, 'source': 'cache'|'bloomberg'|'fallback'|'none'}}
        self.load_log: Dict[str, Dict[str, Any]] = {}

    def _get_default_config_path(self) -> str:
        """Get default path to assets.yaml."""
        return str(Path(__file__).parent.parent.parent / 'config' / 'assets.yaml')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load asset configuration from YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
    
    def _extract_all_tickers(self) -> List[str]:
        """Extract all tickers from configuration."""
        tickers = []
        
        # Rates tickers
        if 'rates' in self.config:
            for country, data in self.config['rates'].items():
                if 'tickers' in data:
                    tickers.extend(data['tickers'])
        
        # FX tickers
        if 'fx' in self.config:
            for currency, data in self.config['fx'].items():
                if 'ticker' in data:
                    tickers.append(data['ticker'])
                    
        # Index tickers
        if 'indices' in self.config:
            for index, data in self.config['indices'].items():
                if 'ticker' in data:
                    tickers.append(data['ticker'])
        
        return list(set(tickers))  # Remove duplicates
    
    def load_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load price data from Bloomberg or cache.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: yesterday)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with price data, indexed by date
        """
        if end_date is None:
            end_date = default_end_date()

        cache_file = self.cache_dir / f"prices_{start_date}_{end_date}.parquet"

        # Try loading from cache
        if use_cache and cache_file.exists():
            logger.info(f"📂 Loading cached data from {cache_file}")
            self.load_log['prices'] = {'file': cache_file, 'source': 'cache'}
            return pd.read_parquet(cache_file)

        # Load from Bloomberg
        if XBBG_AVAILABLE:
            raw = self._load_from_bloomberg(start_date, end_date)
            if raw is not None and not raw.empty:
                df = raw.ffill().dropna()
                if not df.empty:
                    df.to_parquet(cache_file)
                    _write_meta(cache_file, raw)
                    self.load_log['prices'] = {'file': cache_file, 'source': 'bloomberg'}
                    return df
                logger.error("❌ Bloomberg 가격: 결측 없는 행이 없음 (티커 하나가 통째로 비었을 가능성)")

        # Fallback to latest cache
        return self._load_fallback_cache()
    
    def _load_from_bloomberg(
        self,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load data from Bloomberg."""
        try:
            logger.info(f"🔌 Bloomberg 연결 시도 중... ({len(self.all_tickers)} tickers)")
            
            df = blp.bdh(
                tickers=self.all_tickers,
                flds=['px_last'],
                start_date=start_date,
                end_date=end_date,
                Per='D',
                Fill='NA'
            )
            
            if df is None or df.empty:
                raise ValueError("데이터가 비어있습니다.")
            
            # Clean column names (remove MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            
            df.index = pd.to_datetime(df.index)
            # ffill/dropna 는 호출자(load_data)가 메타 기록 후에 한다 — 원자료 필요.
            logger.info(f"✅ Bloomberg 데이터 로드 완료: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Bloomberg 로드 실패: {e}")
            return None
    
    def _load_fallback_cache(self) -> pd.DataFrame:
        """Load most recent cached data as fallback."""
        cache_files = list(self.cache_dir.glob("prices_*.parquet"))
        
        cache_files = [f for f in cache_files if f.name != 'prices_sample.parquet']
        if cache_files:
            latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
            logger.warning(f"⚠️ Bloomberg 실패 — 이전 캐시로 대체: {latest_cache}")
            self.load_log['prices'] = {'file': latest_cache, 'source': 'fallback'}
            return pd.read_parquet(latest_cache)

        # 실매매 경로에서 난수 표본으로 시그널을 내는 일은 없어야 한다.
        raise RuntimeError("가격 데이터 없음 — Bloomberg 실패 + 캐시 없음")
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing without Bloomberg."""
        import numpy as np
        
        dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='B')
        
        data = {}
        np.random.seed(42)
        
        # Sample rates data (yields)
        base_yields = {
            'USGG2YR Index': 2.0,
            'USGG10YR Index': 3.5,
            'GDBR2 Index': 0.5,
            'GDBR10 Index': 1.5,
            'GUKG10 Index': 2.5,
            'GJGB10 Index': 0.5,
            'KE1 Comdty': 108.0,
            'KAA1 Comdty': 120.0,
            'GTAUD3YR Corp': 2.0,
            'GTAUD10YR Corp': 3.0,
            'GFRN10 Index': 1.8,
            'GBTPGR10 Index': 2.5,
        }
        
        for ticker, base in base_yields.items():
            # Random walk with mean reversion
            returns = np.random.randn(len(dates)) * 0.05
            cumulative = np.cumsum(returns)
            mean_reversion = -0.01 * cumulative  # Mean reversion component
            data[ticker] = base + cumulative + mean_reversion
        
        # Sample FX data
        base_fx = {
            'EUR Curncy': 1.10,
            'GBP Curncy': 1.30,
            'JPY Curncy': 110.0,
            'AUD Curncy': 0.70,
            'KRW Curncy': 1200.0,
        }
        
        for ticker, base in base_fx.items():
            vol = 0.005 if 'JPY' not in ticker and 'KRW' not in ticker else 0.003
            returns = np.random.randn(len(dates)) * vol
            data[ticker] = base * np.exp(np.cumsum(returns))
            
        # Sample Equity Index data
        base_indices = {
            'NQ1 Index': 18000.0,
        }
        
        for ticker, base in base_indices.items():
            returns = np.random.randn(len(dates)) * 0.015
            data[ticker] = base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(data, index=dates)
        
        # Save as sample cache
        sample_cache = self.cache_dir / "prices_sample.parquet"
        df.to_parquet(sample_cache)
        logger.info(f"✅ Sample data created: {df.shape}")
        
        return df
    
    def _extract_yield_tickers(self) -> List[str]:
        """Extract signal-only yield tickers from assets.yaml → signal_yields."""
        sy = self.config.get('signal_yields', {}) or {}
        tickers = []
        for t in (sy.get('tradeable_yield_map', {}) or {}).values():
            tickers.append(t)
        for t in (sy.get('fx_short_yield', {}) or {}).values():
            tickers.append(t)
        for t in (sy.get('policy_rate_map', {}) or {}).values():
            tickers.append(t)
        for pair in (sy.get('curve_slope_map', {}) or {}).values():
            tickers.extend(pair)
        for t in (sy.get('extra_yields', []) or []):
            tickers.append(t)
        return sorted(set(tickers))

    def signal_only_tickers(self) -> set:
        """Tickers that exist only as signal inputs (never traded)."""
        return set(self._extract_yield_tickers())

    def merge_signal_yields(self, prices: 'pd.DataFrame') -> 'pd.DataFrame':
        """Append signal-only yield columns to a price panel (aligned, ffilled).

        Yields are loaded once from the long 2010+ cache and reindexed to the
        panel dates, so callers never trigger per-window Bloomberg fetches.
        Returns `prices` unchanged when no yield data is available.
        """
        y = self.load_signal_yields(start_date="2010-01-01", use_cache=True)
        if y is None or y.empty:
            return prices
        y = y.reindex(prices.index).ffill()
        new_cols = [c for c in y.columns if c not in prices.columns]
        if not new_cols:
            return prices
        return prices.join(y[new_cols])

    def load_signal_yields(
        self,
        start_date: str = "2020-01-01",
        end_date: str = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load signal-only yield series (separate from the tradeable panel).

        Returns an empty DataFrame if neither Bloomberg nor a cache is available
        — the SleeveEngine then falls back to price-based Carry/Value proxies.
        """
        yield_tickers = self._extract_yield_tickers()
        if not yield_tickers:
            return pd.DataFrame()

        return self._load_signal_panel('yields', yield_tickers, start_date, end_date, use_cache)

    def _load_signal_panel(self, kind: str, tickers: List[str], start_date: str,
                           end_date: Optional[str], use_cache: bool) -> pd.DataFrame:
        """yields/macro 공용: 캐시 → Bloomberg(+메타) → 이전 캐시 폴백 → 빈 프레임."""
        if end_date is None:
            end_date = default_end_date()
        cache_file = self.cache_dir / f"{kind}_{start_date}_{end_date}.parquet"

        if use_cache and cache_file.exists():
            logger.info(f"📂 Loading cached {kind} from {cache_file}")
            self.load_log[kind] = {'file': cache_file, 'source': 'cache'}
            return pd.read_parquet(cache_file)

        if XBBG_AVAILABLE:
            try:
                raw = blp.bdh(tickers=tickers, flds=['px_last'],
                              start_date=start_date, end_date=end_date, Per='D', Fill='NA')
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = [c[0] for c in raw.columns]
                    raw.index = pd.to_datetime(raw.index)
                    df = raw.ffill()
                    df.to_parquet(cache_file)
                    _write_meta(cache_file, raw)
                    self.load_log[kind] = {'file': cache_file, 'source': 'bloomberg'}
                    logger.info(f"✅ {kind} data loaded: {df.shape}")
                    return df
            except Exception as e:
                logger.error(f"❌ {kind} load failed: {e}")

        caches = list(self.cache_dir.glob(f"{kind}_*.parquet"))
        if caches:
            latest = max(caches, key=lambda x: x.stat().st_mtime)
            logger.warning(f"⚠️ Fallback {kind}: {latest}")
            self.load_log[kind] = {'file': latest, 'source': 'fallback'}
            return pd.read_parquet(latest)
        logger.warning(f"⚠️ No {kind} data available.")
        self.load_log[kind] = {'file': None, 'source': 'none'}
        return pd.DataFrame()

    def _extract_macro_tickers(self) -> List[str]:
        """Extract signal-only macro tickers from assets.yaml → signal_macro."""
        sm = self.config.get('signal_macro', {}) or {}
        tickers = []
        for v in (sm.get('inflation_proxy_map', {}) or {}).values():
            t = v.get('ticker') if isinstance(v, dict) else v
            if t:
                tickers.append(t)
        for pair in (sm.get('policy_gap_map', {}) or {}).values():
            tickers.extend(pair or [])
        for t in (sm.get('carry_short_rate_map', {}) or {}).values():
            if t:
                tickers.append(t)
        tickers.extend(sm.get('extra_macro', []) or [])
        return sorted(set(tickers))

    def load_signal_macro(
        self,
        start_date: str = "2010-01-01",
        end_date: str = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load signal-only macro series (breakevens, OIS, policy rates, CPI YoY).

        Same contract as load_signal_yields: separate cache (macro_*.parquet),
        empty DataFrame when neither Bloomberg nor a cache is available — the
        SleeveEngine's inflation/path sleeves are then silently inactive.
        Monthly/quarterly series (CPI) keep their period-end stamps here; the
        publication-lag shift happens in the engine (macro_pub_lag)."""
        macro_tickers = self._extract_macro_tickers()
        if not macro_tickers:
            return pd.DataFrame()

        return self._load_signal_panel('macro', macro_tickers, start_date, end_date, use_cache)

    # ──────────────────────────────────────────────────────────────────────
    # 연속선물 롤 보정 (2026-10-02)
    # ──────────────────────────────────────────────────────────────────────
    # Bloomberg 제네릭 'XX1 Comdty' 는 무보정 시리즈라 월물 교체일에 근월물→차월물
    # 가격차(≈ 분기 캐리)만큼 가짜 수익률이 찍힌다 (KE1 롤일 잔차 7.8σ, TY1 평균
    # −22bp). 이게 트렌드·실현볼·볼타겟·북스톱 섀도우 손익에 그대로 들어갔다.
    # 보정: 롤일 t 의 수익률을 '새 근월물'의 전일 대비 수익률 P1_t / P2_{t-1} − 1 로
    # 바꾸고, 과거 가격을 그 비율로 뒤에서부터 소급(ratio back-adjust) — 최신
    # 가격은 실제 근월물 가격 그대로라 표시·계약가치 계산은 영향이 없다.
    # 롤일은 FUT_CUR_GEN_TICKER 이력(제네릭이 가리키는 실제 월물)의 변경일.

    def _roll_assets(self) -> List[str]:
        return sorted(t for t in self.all_tickers
                      if 'Comdty' in t and 'NQ' not in t and t.endswith('1 Comdty'))

    def load_roll_info(self, start_date: str = "2010-01-01", end_date: str = None,
                       use_cache: bool = True) -> pd.DataFrame:
        """롤 정보 패널: 'gen::{자산}' (현재 월물 티커) + 'px2::{자산}' (차월물 종가)."""
        if end_date is None:
            end_date = default_end_date()
        cache_file = self.cache_dir / f"rolls_{start_date}_{end_date}.parquet"
        if use_cache and cache_file.exists():
            self.load_log['rolls'] = {'file': cache_file, 'source': 'cache'}
            return pd.read_parquet(cache_file)
        assets = self._roll_assets()
        if XBBG_AVAILABLE and assets:
            try:
                gen = blp.bdh(assets, 'FUT_CUR_GEN_TICKER', start_date, end_date)
                second = {a: a.replace('1 Comdty', '2 Comdty') for a in assets}
                px2 = blp.bdh(list(second.values()), 'PX_LAST', start_date, end_date)
                if gen is not None and not gen.empty and px2 is not None and not px2.empty:
                    for d in (gen, px2):
                        if isinstance(d.columns, pd.MultiIndex):
                            d.columns = [c[0] for c in d.columns]
                        d.index = pd.to_datetime(d.index)
                    inv = {v: k for k, v in second.items()}
                    out = pd.concat([gen.add_prefix('gen::'),
                                     px2.rename(columns=inv).add_prefix('px2::')], axis=1)
                    out.to_parquet(cache_file)
                    _write_meta(cache_file, out)
                    self.load_log['rolls'] = {'file': cache_file, 'source': 'bloomberg'}
                    return out
            except Exception as e:
                logger.error(f"❌ Roll info load failed: {e}")
        caches = list(self.cache_dir.glob("rolls_*.parquet"))
        if caches:
            latest = max(caches, key=lambda x: x.stat().st_mtime)
            logger.warning(f"⚠️ Fallback rolls: {latest}")
            self.load_log['rolls'] = {'file': latest, 'source': 'fallback'}
            return pd.read_parquet(latest)
        self.load_log['rolls'] = {'file': None, 'source': 'none'}
        return pd.DataFrame()

    def roll_adjust(self, prices: pd.DataFrame, info: Optional[pd.DataFrame] = None,
                    max_gap: float = 0.10) -> pd.DataFrame:
        """금리선물 컬럼을 ratio back-adjust (위 설명). 롤 정보가 없으면 원본 + 경고.

        max_gap: |P1/P2 − 1| 가 이보다 큰 롤은 데이터 이상으로 보고 건너뛴다
        (국채선물 월물간 가격차는 통상 1% 미만).
        """
        info = self.load_roll_info() if info is None else info
        if info is None or info.empty:
            logger.warning("⚠️ 롤 정보 없음 — 무보정 연속선물로 진행 (롤 점프 포함)")
            self.load_log['roll_adjust'] = {'applied': [], 'missing': 'all'}
            return prices
        out = prices.copy()
        applied, skipped = [], {}
        for a in out.columns:
            gc, pc = f'gen::{a}', f'px2::{a}'
            if gc not in info.columns or pc not in info.columns:
                continue
            g = info[gc].reindex(out.index).ffill()
            p2 = info[pc].reindex(out.index).ffill()
            roll = (g != g.shift()) & g.notna() & g.shift().notna()
            ratio = out[a].shift() / p2.shift()          # P1_{t-1} / P2_{t-1}
            ok = roll & ratio.notna() & ((ratio - 1.0).abs() <= max_gap)
            bad = int((roll & ~ok).sum())
            if bad:
                skipped[a] = bad
            m = pd.Series(1.0, index=out.index)
            m[ok] = 1.0 / ratio[ok]
            # K_t = Π_{롤일 s > t} (P2_{s-1}/P1_{s-1}) — 최신 K=1 (실제 가격 유지)
            K = m[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
            out[a] = out[a] * K
            applied.append(a)
        if skipped:
            logger.warning(f"⚠️ 롤 보정 일부 건너뜀 (가격차 > {max_gap:.0%}): {skipped}")
        self.load_log['roll_adjust'] = {'applied': applied, 'skipped': skipped}
        return out

    def engine_prices(self, cfg: Optional[Dict[str, Any]] = None,
                      start_date: str = "2010-01-01") -> pd.DataFrame:
        """슬리브 엔진 입력 가격 패널의 단일 경로: 캐시 로드 → ffill/dropna →
        (cfg.roll_adjust 면) 롤 보정. 백테스트·대시보드·감사가 모두 이걸 써야
        서로 다른 가격을 보는 일이 없다."""
        from .preprocessor import DataPreprocessor
        px = DataPreprocessor(self.load_data(start_date=start_date, use_cache=True)).clean().get_data()
        if (cfg or {}).get('roll_adjust', False):
            px = self.roll_adjust(px)
        return px

    # ──────────────────────────────────────────────────────────────────────
    # 데이터 신선도 점검 (2026-10-02)
    # ──────────────────────────────────────────────────────────────────────
    def freshness_report(self, watch: Optional[List[str]] = None,
                         end_date: Optional[str] = None) -> Dict[str, Any]:
        """이 로더가 읽은 캐시의 신선도 점검 → {'data_asof', 'expected', 'issues': [...]}.

        issues 항목은 사람이 읽는 경고 문장. 비어 있으면 정상.
          · 이전 캐시 폴백(Bloomberg 실패) — 가장 심각: 시그널이 '오늘 것'이 아님
          · 패널 마지막 날짜 < 기대 세션 (평일 기준; 양 시장 공휴일이면 정상)
          · 감시 티커의 마지막 '실데이터' < 패널 마지막 날짜 → ffill 로 전일값 사용 중
            (해당 시장 휴장일이면 정상, 아니면 Bloomberg 누락)
          · 롤 정보가 가격보다 오래됨 → 최근 롤이 보정 안 된 채 시그널에 들어감
        """
        expected = expected_last_session(end_date)
        issues: List[str] = []
        asof = {}
        for kind in ('prices', 'yields', 'macro', 'rolls'):
            ent = self.load_log.get(kind)
            if not ent or not ent.get('file'):
                continue
            f = Path(ent['file'])
            try:
                last = pd.read_parquet(f).index.max()
            except Exception:
                continue
            asof[kind] = last
            if ent.get('source') == 'fallback':
                issues.append(f"[{kind}] Bloomberg 실패 → 이전 캐시 사용 ({f.name}, 마지막 {last.date()})")
            if last < expected and kind in ('prices', 'yields'):
                issues.append(f"[{kind}] 마지막 날짜 {last.date()} < 기대 세션 {expected.date()} "
                              f"(양 시장 공휴일이 아니면 데이터가 오래됨)")
            lv = _read_meta(f).get('last_valid', {})
            for t in (watch or []):
                d = lv.get(t)
                if d is None:
                    continue
                if pd.Timestamp(d) < last:
                    issues.append(f"[{kind}] {t} 실데이터 마지막 {d} < 패널 {last.date()} "
                                  f"— 휴장이 아니면 Bloomberg 누락 (전일값으로 채워짐)")
        if 'rolls' in asof and 'prices' in asof and asof['rolls'] < asof['prices']:
            issues.append(f"[rolls] 롤 정보 {asof['rolls'].date()} < 가격 {asof['prices'].date()} "
                          f"— 그 사이 롤이 있었다면 보정되지 않음")
        ra = self.load_log.get('roll_adjust')
        if ra and ra.get('missing') == 'all':
            issues.append("[rolls] 롤 정보 없음 — 무보정 연속선물로 시그널 계산 중")
        data_asof = min((asof[k] for k in ('prices', 'yields') if k in asof), default=None)
        return {'data_asof': None if data_asof is None else str(data_asof.date()),
                'expected': str(expected.date()), 'issues': issues}

    def get_cross_asset_pairs(self) -> Dict[str, List[tuple]]:
        """Get cross-asset pairs from configuration."""
        return self.config.get('cross_asset_pairs', {})
    
    def get_tickers_by_category(self) -> Dict[str, List[str]]:
        """Get tickers organized by category (rates/fx)."""
        result = {'rates': [], 'fx': []}
        
        if 'rates' in self.config:
            for country, data in self.config['rates'].items():
                if 'tickers' in data:
                    result['rates'].extend(data['tickers'])
        
        if 'fx' in self.config:
            for currency, data in self.config['fx'].items():
                if 'ticker' in data:
                    result['fx'].append(data['ticker'])
                    
        if 'indices' in self.config:
            result['indices'] = []
            for index, data in self.config['indices'].items():
                if 'ticker' in data:
                    result['indices'].append(data['ticker'])
        
        return result
