"""SQLite persistence for the paper account.

One file holds everything: cash, positions, orders, the trade journal, daily
equity, every strategy the bot has used, and a log of each learning decision.
That makes the account inspectable with any SQLite tool, and easy to reset.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evotrader.evolution.genome import Genome

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY, qty REAL NOT NULL, avg_price REAL NOT NULL,
    entry_date TEXT NOT NULL, entry_reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, created TEXT NOT NULL, ticker TEXT NOT NULL, side TEXT NOT NULL,
    qty REAL NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL,
    fill_date TEXT, fill_price REAL, cost REAL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, strategy_id INTEGER NOT NULL,
    entry_date TEXT NOT NULL, entry_price REAL NOT NULL, exit_date TEXT NOT NULL,
    exit_price REAL NOT NULL, qty REAL NOT NULL, pnl REAL NOT NULL, return REAL NOT NULL,
    entry_reason TEXT NOT NULL, exit_reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS equity (
    date TEXT PRIMARY KEY, cash REAL NOT NULL, holdings REAL NOT NULL, equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY, created TEXT NOT NULL, rule TEXT NOT NULL, genome TEXT NOT NULL,
    origin TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS learning_log (
    id INTEGER PRIMARY KEY, date TEXT NOT NULL, incumbent_score REAL, challenger_score REAL,
    challenger_rule TEXT, promoted INTEGER NOT NULL, notes TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Position:
    ticker: str
    qty: float
    avg_price: float
    entry_date: str
    entry_reason: str


@dataclass(frozen=True)
class Order:
    id: int
    created: str
    ticker: str
    side: str  # "buy" | "sell"
    qty: float
    reason: str


class PaperStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def commit(self) -> None:
        self.db.commit()

    # --- meta ---------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    @property
    def initialised(self) -> bool:
        return self.get("initial_capital") is not None

    @property
    def cash(self) -> float:
        return float(self.get("cash", 0.0))

    @cash.setter
    def cash(self, value: float) -> None:
        self.set("cash", value)

    # --- positions & orders -------------------------------------------------
    def positions(self) -> dict[str, Position]:
        rows = self.db.execute("SELECT * FROM positions").fetchall()
        return {r["ticker"]: Position(**dict(r)) for r in rows}

    def upsert_position(self, pos: Position) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?, ?)",
            (pos.ticker, pos.qty, pos.avg_price, pos.entry_date, pos.entry_reason),
        )

    def delete_position(self, ticker: str) -> None:
        self.db.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))

    def add_order(self, created: str, ticker: str, side: str, qty: float, reason: str) -> None:
        self.db.execute(
            "INSERT INTO orders (created, ticker, side, qty, status, reason) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (created, ticker, side, qty, reason),
        )

    def pending_orders(self) -> list[Order]:
        rows = self.db.execute(
            "SELECT id, created, ticker, side, qty, reason FROM orders "
            "WHERE status = 'pending' ORDER BY id"
        ).fetchall()
        return [Order(**dict(r)) for r in rows]

    def mark_filled(self, order_id: int, date: str, price: float, cost: float) -> None:
        self.db.execute(
            "UPDATE orders SET status = 'filled', fill_date = ?, fill_price = ?, cost = ? "
            "WHERE id = ?",
            (date, price, cost, order_id),
        )

    # --- journal ------------------------------------------------------------
    def add_trade(self, **fields: Any) -> None:
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        self.db.execute(f"INSERT INTO trades ({cols}) VALUES ({marks})", tuple(fields.values()))

    def record_equity(self, date: str, cash: float, holdings: float) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO equity VALUES (?, ?, ?, ?)", (date, cash, holdings,
                                                               cash + holdings)
        )

    def log_learning(self, date: str, incumbent_score: float | None,
                     challenger_score: float | None, challenger_rule: str | None,
                     promoted: bool, notes: str) -> None:
        self.db.execute(
            "INSERT INTO learning_log (date, incumbent_score, challenger_score, challenger_rule,"
            " promoted, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (date, incumbent_score, challenger_score, challenger_rule, int(promoted), notes),
        )

    # --- strategies ---------------------------------------------------------
    def activate_strategy(self, genome: Genome, created: str, origin: str) -> int:
        self.db.execute("UPDATE strategies SET active = 0")
        cur = self.db.execute(
            "INSERT INTO strategies (created, rule, genome, origin, active) VALUES (?, ?, ?, ?, 1)",
            (created, str(genome), json.dumps(genome.to_dict()), origin),
        )
        return int(cur.lastrowid)

    def active_strategy(self) -> tuple[int, Genome]:
        row = self.db.execute("SELECT id, genome FROM strategies WHERE active = 1").fetchone()
        if row is None:
            raise RuntimeError("No active strategy; run `evotrader paper init` first")
        return int(row["id"]), Genome.from_dict(json.loads(row["genome"]))

    def hall_of_fame(self) -> list[Genome]:
        return [Genome.from_dict(d) for d in self.get("hall_of_fame", [])]

    def set_hall_of_fame(self, genomes: list[Genome]) -> None:
        self.set("hall_of_fame", [g.to_dict() for g in genomes])

    # --- reading for reports ------------------------------------------------
    def frame(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.db, params=params)
