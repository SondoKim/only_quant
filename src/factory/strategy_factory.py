"""
Strategy Factory for Global Macro Trading

Manages strategy storage, retrieval, and lifecycle in SQLite.
"""

import json
import logging
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id   TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    asset         TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    params        TEXT NOT NULL,          -- JSON string
    related_asset TEXT,
    sharpe_3y     REAL DEFAULT 0,
    sharpe_6m     REAL DEFAULT 0,
    sharpe_1m     REAL DEFAULT 0,
    total_return  REAL DEFAULT 0,
    annual_return REAL DEFAULT 0,
    max_drawdown  REAL DEFAULT 0,
    win_rate      REAL DEFAULT 0,
    num_trades    INTEGER DEFAULT 0,
    calmar_ratio  REAL DEFAULT 0,
    sortino_ratio REAL DEFAULT 0,
    is_active     INTEGER DEFAULT 0,
    version       INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_active     ON strategies(is_active);
CREATE INDEX IF NOT EXISTS idx_sharpe_3y  ON strategies(sharpe_3y);
CREATE INDEX IF NOT EXISTS idx_sharpe_6m  ON strategies(sharpe_6m);
CREATE INDEX IF NOT EXISTS idx_asset      ON strategies(asset);
CREATE INDEX IF NOT EXISTS idx_type       ON strategies(strategy_type);
"""


class StrategyFactory:
    """Manage strategy storage and retrieval using SQLite."""

    def __init__(self, storage_dir: str = None):
        if not storage_dir:
            raise ValueError(
                "storage_dir must be provided. "
                "Use main.py --mode discover/signals to auto-resolve."
            )
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.storage_dir / 'strategies.db'
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

        # Legacy compatibility: expose index dict for run_backtest.py
        # that reads factory.index['strategies']
        self.index = self._build_index_compat()

    # ------------------------------------------------------------------
    # DB lifecycle
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=10)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Legacy index compatibility
    # ------------------------------------------------------------------
    def _build_index_compat(self) -> Dict[str, Any]:
        """Build a dict matching the old index.json structure."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT strategy_id, asset, strategy_type, "
            "       sharpe_3y, sharpe_6m, sortino_ratio, is_active, updated_at "
            "FROM strategies"
        ).fetchall()
        strategies = {}
        last_updated = None
        for r in rows:
            strategies[r['strategy_id']] = {
                'file': f"{r['strategy_id']}.json",
                'asset': r['asset'],
                'strategy_type': r['strategy_type'],
                'sharpe_3y': r['sharpe_3y'],
                'sharpe_6m': r['sharpe_6m'],
                'sortino_6m': r['sortino_ratio'],
                'is_active': bool(r['is_active']),
                'updated_at': r['updated_at'],
            }
            if last_updated is None or r['updated_at'] > (last_updated or ''):
                last_updated = r['updated_at']
        return {'strategies': strategies, 'last_updated': last_updated}

    def _save_index(self):
        """Refresh the in-memory index cache. No file I/O needed."""
        self.index = self._build_index_compat()

    # ------------------------------------------------------------------
    # Row ↔ Dict conversion
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a DB row to the legacy strategy dict format."""
        return {
            'strategy_id': row['strategy_id'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
            'asset': row['asset'],
            'strategy_type': row['strategy_type'],
            'strategy_name': row['strategy_name'],
            'params': json.loads(row['params']),
            'related_asset': row['related_asset'],
            'performance': {
                'sharpe_3y': row['sharpe_3y'],
                'sharpe_6m': row['sharpe_6m'],
                'sharpe_1m': row['sharpe_1m'],
                'total_return': row['total_return'],
                'annual_return': row['annual_return'],
                'max_drawdown': row['max_drawdown'],
                'win_rate': row['win_rate'],
                'num_trades': row['num_trades'],
                'calmar_ratio': row['calmar_ratio'],
                'sortino_ratio': row['sortino_ratio'],
            },
            'is_active': bool(row['is_active']),
            'version': row['version'],
        }

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------
    def _generate_strategy_id(self, strategy_config: Dict[str, Any]) -> str:
        key_fields = {
            'asset': strategy_config.get('asset'),
            'strategy_type': strategy_config.get('strategy_type'),
            'strategy_name': strategy_config.get('strategy_name'),
            'params': str(sorted(strategy_config.get('params', {}).items())),
            'related_asset': strategy_config.get('related_asset'),
        }
        hash_str = json.dumps(key_fields, sort_keys=True)
        hash_id = hashlib.md5(hash_str.encode()).hexdigest()[:8]

        asset_short = strategy_config.get('asset', 'UNK')[:6]
        strategy_short = strategy_config.get('strategy_name', 'UNK')[:4].upper()
        type_short = 'MOM' if strategy_config.get('strategy_type') == 'momentum' else 'MR'

        return f"{asset_short}_{type_short}_{strategy_short}_{hash_id}"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def save_strategy(
        self,
        strategy_config: Dict[str, Any],
        performance: Dict[str, Any]
    ) -> str:
        strategy_id = strategy_config.get('id') or self._generate_strategy_id(strategy_config)
        now = datetime.now().isoformat()
        params_json = json.dumps(strategy_config.get('params', {}), sort_keys=True)

        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO strategies
               (strategy_id, created_at, updated_at, asset, strategy_type,
                strategy_name, params, related_asset,
                sharpe_3y, sharpe_6m, sharpe_1m, total_return, annual_return,
                max_drawdown, win_rate, num_trades, calmar_ratio, sortino_ratio,
                is_active, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,1)""",
            (
                strategy_id, now, now,
                strategy_config.get('asset'),
                strategy_config.get('strategy_type'),
                strategy_config.get('strategy_name'),
                params_json,
                strategy_config.get('related_asset'),
                float(performance.get('sharpe_3y', 0)),
                float(performance.get('sharpe_6m', 0)),
                float(performance.get('sharpe_1m', 0)),
                float(performance.get('total_return', 0)),
                float(performance.get('annual_return', 0)),
                float(performance.get('max_drawdown', 0)),
                float(performance.get('win_rate', 0)),
                int(performance.get('num_trades', 0)),
                float(performance.get('calmar_ratio', 0)),
                float(performance.get('sortino_ratio', 0)),
            ),
        )
        conn.commit()

        # Update in-memory index
        self.index['strategies'][strategy_id] = {
            'file': f"{strategy_id}.json",
            'asset': strategy_config.get('asset'),
            'strategy_type': strategy_config.get('strategy_type'),
            'sharpe_3y': float(performance.get('sharpe_3y', 0)),
            'sharpe_6m': float(performance.get('sharpe_6m', 0)),
            'sortino_6m': float(performance.get('sortino_ratio', 0)),
            'is_active': False,
            'updated_at': now,
        }
        self.index['last_updated'] = now

        logger.info(f"Strategy saved: {strategy_id}")
        return strategy_id

    def save_strategies_batch(
        self,
        items: List[tuple],
    ) -> int:
        """Batch insert strategies. Each item is (strategy_config, performance).
        Returns count of inserted rows."""
        now = datetime.now().isoformat()
        rows = []
        for strategy_config, performance in items:
            strategy_id = strategy_config.get('id') or self._generate_strategy_id(strategy_config)
            params_json = json.dumps(strategy_config.get('params', {}), sort_keys=True)
            rows.append((
                strategy_id, now, now,
                strategy_config.get('asset'),
                strategy_config.get('strategy_type'),
                strategy_config.get('strategy_name'),
                params_json,
                strategy_config.get('related_asset'),
                float(performance.get('sharpe_3y', 0)),
                float(performance.get('sharpe_6m', 0)),
                float(performance.get('sharpe_1m', 0)),
                float(performance.get('total_return', 0)),
                float(performance.get('annual_return', 0)),
                float(performance.get('max_drawdown', 0)),
                float(performance.get('win_rate', 0)),
                int(performance.get('num_trades', 0)),
                float(performance.get('calmar_ratio', 0)),
                float(performance.get('sortino_ratio', 0)),
            ))

        conn = self._get_conn()
        conn.executemany(
            """INSERT OR REPLACE INTO strategies
               (strategy_id, created_at, updated_at, asset, strategy_type,
                strategy_name, params, related_asset,
                sharpe_3y, sharpe_6m, sharpe_1m, total_return, annual_return,
                max_drawdown, win_rate, num_trades, calmar_ratio, sortino_ratio,
                is_active, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,1)""",
            rows,
        )
        conn.commit()
        self._save_index()
        return len(rows)

    def load_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if row is None:
            logger.warning(f"Strategy not found: {strategy_id}")
            return None
        return self._row_to_dict(row)

    def update_performance(
        self,
        strategy_id: str,
        performance: Dict[str, Any]
    ) -> bool:
        # Clean values
        def _clean(v):
            if hasattr(v, 'tolist'):
                val = v.tolist()
                return val[0] if isinstance(val, list) and len(val) == 1 else val
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        perf = {k: _clean(v) for k, v in performance.items()}
        now = datetime.now().isoformat()

        conn = self._get_conn()
        existing = conn.execute(
            "SELECT version FROM strategies WHERE strategy_id = ?", (strategy_id,)
        ).fetchone()
        if existing is None:
            return False

        conn.execute(
            """UPDATE strategies SET
                sharpe_3y=?, sharpe_6m=?, sharpe_1m=?,
                total_return=?, annual_return=?, max_drawdown=?,
                win_rate=?, num_trades=?, calmar_ratio=?, sortino_ratio=?,
                updated_at=?, version=?
               WHERE strategy_id=?""",
            (
                float(perf.get('sharpe_3y', 0)),
                float(perf.get('sharpe_6m', 0)),
                float(perf.get('sharpe_1m', 0)),
                float(perf.get('total_return', 0)),
                float(perf.get('annual_return', 0)),
                float(perf.get('max_drawdown', 0)),
                float(perf.get('win_rate', 0)),
                int(perf.get('num_trades', 0)),
                float(perf.get('calmar_ratio', 0)),
                float(perf.get('sortino_ratio', 0)),
                now,
                existing['version'] + 1,
                strategy_id,
            ),
        )
        conn.commit()

        # Sync in-memory index
        if strategy_id in self.index['strategies']:
            self.index['strategies'][strategy_id].update({
                'sharpe_3y': float(perf.get('sharpe_3y', self.index['strategies'][strategy_id]['sharpe_3y'])),
                'sharpe_6m': float(perf.get('sharpe_6m', self.index['strategies'][strategy_id]['sharpe_6m'])),
                'updated_at': now,
            })
        return True

    def set_active(self, strategy_id: str, active: bool = True) -> bool:
        now = datetime.now().isoformat()
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE strategies SET is_active=?, updated_at=? WHERE strategy_id=?",
            (int(active), now, strategy_id),
        )
        if cur.rowcount == 0:
            return False
        conn.commit()

        if strategy_id in self.index['strategies']:
            self.index['strategies'][strategy_id]['is_active'] = active
        return True

    def set_all_active(self, active: bool = True) -> int:
        """Bulk set all strategies active/inactive. Returns affected count."""
        now = datetime.now().isoformat()
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE strategies SET is_active=?, updated_at=?",
            (int(active), now),
        )
        conn.commit()
        for info in self.index['strategies'].values():
            info['is_active'] = active
        return cur.rowcount

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def filter_by_sharpe_3y(self, min_sharpe: float = 0.8) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM strategies WHERE sharpe_3y >= ?", (min_sharpe,)
        ).fetchall()
        result = [self._row_to_dict(r) for r in rows]
        logger.info(f"Found {len(result)} strategies with Sharpe 3Y >= {min_sharpe}")
        return result

    def filter_by_sharpe_6m(self, min_sharpe: float = 1.1) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM strategies WHERE sharpe_6m >= ?", (min_sharpe,)
        ).fetchall()
        result = [self._row_to_dict(r) for r in rows]
        logger.info(f"Found {len(result)} strategies with Sharpe 6M >= {min_sharpe}")
        return result

    def get_active_strategies(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM strategies WHERE is_active = 1"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def activate_qualified_strategies(
        self,
        sharpe_6m_min: float = 0.5,
        sortino_6m_min: float = 0.5
    ) -> List[str]:
        now = datetime.now().isoformat()
        conn = self._get_conn()

        # Activate qualifying
        conn.execute(
            """UPDATE strategies SET is_active=1, updated_at=?
               WHERE sharpe_6m >= ? AND sortino_ratio >= ? AND is_active=0""",
            (now, sharpe_6m_min, sortino_6m_min),
        )
        # Deactivate non-qualifying
        conn.execute(
            """UPDATE strategies SET is_active=0, updated_at=?
               WHERE (sharpe_6m < ? OR sortino_ratio < ?) AND is_active=1""",
            (now, sharpe_6m_min, sortino_6m_min),
        )
        conn.commit()

        activated = [
            r['strategy_id'] for r in conn.execute(
                "SELECT strategy_id FROM strategies WHERE is_active=1"
            ).fetchall()
        ]
        self._save_index()
        logger.info(f"Activated {len(activated)} strategies (Sharpe 6M >= {sharpe_6m_min}, Sortino 6M >= {sortino_6m_min})")
        return activated

    def get_summary(self) -> Dict[str, Any]:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) c FROM strategies").fetchone()['c']
        active = conn.execute("SELECT COUNT(*) c FROM strategies WHERE is_active=1").fetchone()['c']
        by_type = {'momentum': 0, 'mean_reversion': 0, 'advanced': 0}
        for row in conn.execute("SELECT strategy_type, COUNT(*) c FROM strategies GROUP BY strategy_type"):
            if row['strategy_type'] in by_type:
                by_type[row['strategy_type']] = row['c']

        return {
            'total_strategies': total,
            'active_strategies': active,
            'by_type': by_type,
            'last_updated': self.index.get('last_updated'),
        }

    def delete_strategy(self, strategy_id: str) -> bool:
        conn = self._get_conn()
        conn.execute("DELETE FROM strategies WHERE strategy_id=?", (strategy_id,))
        conn.commit()
        self.index['strategies'].pop(strategy_id, None)
        logger.info(f"Strategy deleted: {strategy_id}")
        return True

    def cleanup_underperformers(
        self,
        sharpe_3y_min: float = 0.5,
        max_age_days: int = 365
    ) -> int:
        conn = self._get_conn()
        cutoff = datetime.now()

        to_delete = []
        for row in conn.execute("SELECT strategy_id, sharpe_3y, is_active, updated_at FROM strategies"):
            if row['sharpe_3y'] < sharpe_3y_min:
                to_delete.append(row['strategy_id'])
                continue
            if row['updated_at']:
                age = (cutoff - datetime.fromisoformat(row['updated_at'])).days
                if age > max_age_days and not row['is_active']:
                    to_delete.append(row['strategy_id'])

        if to_delete:
            conn.executemany(
                "DELETE FROM strategies WHERE strategy_id=?",
                [(sid,) for sid in to_delete],
            )
            conn.commit()
            for sid in to_delete:
                self.index['strategies'].pop(sid, None)

        logger.info(f"Cleaned up {len(to_delete)} underperforming strategies")
        return len(to_delete)

    def get_all_strategy_configs(self) -> List[Dict[str, Any]]:
        """Get all strategy configs (for incremental discovery).
        Returns list of dicts with keys matching generator output format."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT strategy_id, asset, strategy_type, strategy_name, "
            "       params, related_asset, sharpe_3y "
            "FROM strategies"
        ).fetchall()
        result = []
        for r in rows:
            result.append({
                'id': r['strategy_id'],
                'asset': r['asset'],
                'strategy_type': r['strategy_type'],
                'strategy_name': r['strategy_name'],
                'params': json.loads(r['params']),
                'related_asset': r['related_asset'],
                '_prev_sharpe_3y': r['sharpe_3y'],
            })
        return result

    def strategy_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) c FROM strategies").fetchone()['c']
