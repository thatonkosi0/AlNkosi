"""SQLite-backed strategy vault and campaign store."""

from __future__ import annotations

import io
import json
import sqlite3
import zlib
from pathlib import Path

import numpy as np

_METRIC_KEYS = [
    "net_profit",
    "max_drawdown",
    "profit_factor",
    "sharpe",
    "win_rate",
    "trades",
    "avg_yearly_return",
]

_SORTABLE = {f"{p}_{k}" for p in ("is", "oos") for k in _METRIC_KEYS} | {
    "id",
    "created_at",
    "name",
    "tribe",
    "symbol",
    "timeframe",
}

# Metric-range filters accepted by list_strategies: name -> (column, operator).
_RANGE_FILTERS = {
    "min_oos_net_profit": ("oos_net_profit", ">="),
    "min_oos_profit_factor": ("oos_profit_factor", ">="),
    "min_oos_sharpe": ("oos_sharpe", ">="),
    "min_oos_trades": ("oos_trades", ">="),
    "max_oos_max_drawdown": ("oos_max_drawdown", "<="),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    tribe TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    genome_json TEXT NOT NULL,
    is_json TEXT NOT NULL,
    oos_json TEXT NOT NULL,
    equity_blob BLOB,
    campaign_id INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    {metric_cols}
);
CREATE TABLE IF NOT EXISTS campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    progress_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
""".format(
    metric_cols=",\n    ".join(
        f"{p}_{k} REAL" for p in ("is", "oos") for k in _METRIC_KEYS
    )
)


def _pack_equity(equity: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(equity, dtype=np.float32))
    return zlib.compress(buf.getvalue())


def _unpack_equity(blob: bytes) -> list[float]:
    arr = np.load(io.BytesIO(zlib.decompress(blob)))
    return [float(x) for x in arr]


class Vault:
    def __init__(self, db_path: Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    # -- strategies -------------------------------------------------------

    def add_strategy(
        self,
        *,
        name: str | None,
        tribe: str,
        symbol: str,
        timeframe: str,
        genome_json: str,
        is_metrics: dict,
        oos_metrics: dict,
        equity: np.ndarray,
        campaign_id: int | None,
    ) -> int:
        cols = ["name", "tribe", "symbol", "timeframe", "genome_json", "is_json",
                "oos_json", "equity_blob", "campaign_id"]
        vals = [name, tribe, symbol, timeframe, genome_json, json.dumps(is_metrics),
                json.dumps(oos_metrics), _pack_equity(equity), campaign_id]
        for prefix, metrics in (("is", is_metrics), ("oos", oos_metrics)):
            for key in _METRIC_KEYS:
                cols.append(f"{prefix}_{key}")
                vals.append(metrics.get(key))
        placeholders = ", ".join("?" for _ in cols)
        with self._conn() as con:
            cur = con.execute(
                f"INSERT INTO strategies ({', '.join(cols)}) VALUES ({placeholders})",
                vals,
            )
            sid = cur.lastrowid
            if name is None:
                auto = f"ALG-{tribe[:2].upper()}-{sid:05d}"
                con.execute("UPDATE strategies SET name = ? WHERE id = ?", (auto, sid))
        return sid

    def list_strategies(
        self,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        tribe: str | None = None,
        sort: str = "oos_net_profit",
        desc: bool = True,
        limit: int = 200,
        **ranges: float | None,
    ) -> list[dict]:
        if sort not in _SORTABLE:
            raise ValueError(f"Unknown sort column {sort!r}; allowed: {sorted(_SORTABLE)}")
        unknown = set(ranges) - set(_RANGE_FILTERS)
        if unknown:
            raise ValueError(
                f"Unknown metric filters {sorted(unknown)}; allowed: {sorted(_RANGE_FILTERS)}"
            )
        where, params = [], []
        for col, val in (("symbol", symbol), ("timeframe", timeframe), ("tribe", tribe)):
            if val is not None:
                where.append(f"{col} = ?")
                params.append(val)
        for name, val in ranges.items():
            if val is not None:
                col, op = _RANGE_FILTERS[name]
                where.append(f"{col} {op} ?")
                params.append(float(val))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        order = "DESC" if desc else "ASC"
        sortable_cols = ", ".join(
            f"{p}_{k}" for p in ("is", "oos") for k in _METRIC_KEYS
        )
        with self._conn() as con:
            rows = con.execute(
                f"SELECT id, name, tribe, symbol, timeframe, campaign_id, created_at, "
                f"{sortable_cols} FROM strategies {clause} "
                f"ORDER BY {sort} {order} LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_strategy(self, sid: int) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM strategies WHERE id = ?", (sid,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["is_metrics"] = json.loads(d.pop("is_json"))
        d["oos_metrics"] = json.loads(d.pop("oos_json"))
        blob = d.pop("equity_blob")
        d["equity"] = _unpack_equity(blob) if blob else []
        return d

    def campaign_tribe_genomes(self, campaign_id: int, symbol: str, tribe: str) -> set[str]:
        """Genome JSONs already vaulted by a campaign for one symbol/tribe.

        Seeds the GA's dedupe set when a crashed campaign resumes, so the
        rerun never writes the same strategy twice.
        """
        with self._conn() as con:
            rows = con.execute(
                "SELECT genome_json FROM strategies "
                "WHERE campaign_id = ? AND symbol = ? AND tribe = ?",
                (campaign_id, symbol, tribe),
            ).fetchall()
        return {r["genome_json"] for r in rows}

    def delete_strategy(self, sid: int) -> bool:
        with self._conn() as con:
            cur = con.execute("DELETE FROM strategies WHERE id = ?", (sid,))
        return cur.rowcount > 0

    # -- campaigns --------------------------------------------------------

    def create_campaign(self, config_json: str) -> int:
        with self._conn() as con:
            cur = con.execute(
                "INSERT INTO campaigns (config_json) VALUES (?)", (config_json,)
            )
        return cur.lastrowid

    def update_campaign(
        self, cid: int, *, status: str | None = None, progress_json: str | None = None
    ) -> None:
        sets, params = ["updated_at = datetime('now')"], []
        if status is not None:
            sets.append("status = ?")
            params.append(status)
        if progress_json is not None:
            sets.append("progress_json = ?")
            params.append(progress_json)
        with self._conn() as con:
            con.execute(
                f"UPDATE campaigns SET {', '.join(sets)} WHERE id = ?", (*params, cid)
            )

    def get_campaign(self, cid: int) -> dict | None:
        with self._conn() as con:
            row = con.execute("SELECT * FROM campaigns WHERE id = ?", (cid,)).fetchone()
        return dict(row) if row else None

    def list_campaigns(self, limit: int = 50) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM campaigns ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- analytics --------------------------------------------------------

    def insights(self) -> dict:
        out = {}
        for name, col in (("by_symbol", "symbol"), ("by_timeframe", "timeframe"), ("by_tribe", "tribe")):
            with self._conn() as con:
                rows = con.execute(
                    f"SELECT {col} AS key, COUNT(*) AS count, "
                    f"AVG(oos_net_profit) AS avg_oos_net_profit, "
                    f"AVG(oos_profit_factor) AS avg_oos_profit_factor "
                    f"FROM strategies GROUP BY {col} ORDER BY avg_oos_net_profit DESC"
                ).fetchall()
            out[name] = [dict(r) for r in rows]
        return out

    def count(self) -> int:
        with self._conn() as con:
            return con.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]
