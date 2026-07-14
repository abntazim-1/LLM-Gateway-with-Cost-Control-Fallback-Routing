import sqlite3
from typing import Dict, Any, Optional
import os
import threading

class LedgerStore:
    """A simple thread-safe SQLite store for budgets and costs."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = "ledger.db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LedgerStore, cls).__new__(cls)
                cls._instance._init_db(db_path)
        return cls._instance

    def _init_db(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    api_key TEXT PRIMARY KEY,
                    daily_limit_usd REAL,
                    monthly_limit_usd REAL,
                    spend_today REAL DEFAULT 0.0,
                    spend_month REAL DEFAULT 0.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY,
                    api_key TEXT,
                    backend TEXT,
                    model TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    cost_usd REAL,
                    latency_ms REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def load_budgets_from_config(self, budgets_config: list):
        """Seed initial budgets from config."""
        with sqlite3.connect(self.db_path) as conn:
            for b in budgets_config:
                conn.execute("""
                    INSERT INTO budgets (api_key, daily_limit_usd, monthly_limit_usd)
                    VALUES (?, ?, ?)
                    ON CONFLICT(api_key) DO UPDATE SET
                    daily_limit_usd=excluded.daily_limit_usd,
                    monthly_limit_usd=excluded.monthly_limit_usd
                """, (b["api_key"], b["daily_limit_usd"], b["monthly_limit_usd"]))
            conn.commit()

    def get_budget(self, api_key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM budgets WHERE api_key = ?", (api_key,)).fetchone()
            return dict(row) if row else None

    def record_request(self, api_key: str, req_id: str, backend: str, model: str, 
                       prompt_tokens: int, comp_tokens: int, cost: float, latency: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO requests (id, api_key, backend, model, prompt_tokens, completion_tokens, cost_usd, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (req_id, api_key, backend, model, prompt_tokens, comp_tokens, cost, latency))
            
            # Update budget
            conn.execute("""
                UPDATE budgets 
                SET spend_today = spend_today + ?, spend_month = spend_month + ?
                WHERE api_key = ?
            """, (cost, cost, api_key))
            conn.commit()
