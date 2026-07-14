import sqlite3
from typing import Dict, Any, Optional
import os
from datetime import datetime

class LedgerStore:
    """A thread-safe SQLite store for budgets and costs."""

    def __init__(self, db_path: str = "ledger.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    api_key TEXT PRIMARY KEY,
                    daily_limit_usd REAL,
                    monthly_limit_usd REAL,
                    spend_today REAL DEFAULT 0.0,
                    spend_month REAL DEFAULT 0.0,
                    last_reset_date TEXT,
                    last_reset_month TEXT
                )
            """)
            
            # safely migrate existing DB
            try:
                conn.execute("ALTER TABLE budgets ADD COLUMN last_reset_date TEXT")
                conn.execute("ALTER TABLE budgets ADD COLUMN last_reset_month TEXT")
            except sqlite3.OperationalError:
                pass # columns already exist
                
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breakers (
                    backend_id TEXT PRIMARY KEY,
                    state TEXT,
                    consecutive_failures INTEGER,
                    last_failure_time REAL
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
            if not row:
                return None
                
            budget = dict(row)
            today_str = datetime.utcnow().strftime('%Y-%m-%d')
            month_str = datetime.utcnow().strftime('%Y-%m')
            
            needs_update = False
            
            # Reset daily spend if a new day has started
            if budget.get('last_reset_date') != today_str:
                budget['spend_today'] = 0.0
                budget['last_reset_date'] = today_str
                needs_update = True
                
            # Reset monthly spend if a new month has started
            if budget.get('last_reset_month') != month_str:
                budget['spend_month'] = 0.0
                budget['last_reset_month'] = month_str
                needs_update = True
                
            if needs_update:
                conn.execute("""
                    UPDATE budgets 
                    SET spend_today = ?, last_reset_date = ?, 
                        spend_month = ?, last_reset_month = ?
                    WHERE api_key = ?
                """, (budget['spend_today'], budget['last_reset_date'],
                      budget['spend_month'], budget['last_reset_month'], api_key))
                conn.commit()
                
            return budget

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

    def get_circuit_breaker_state(self, backend_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM circuit_breakers WHERE backend_id = ?", (backend_id,)).fetchone()
            return dict(row) if row else None

    def update_circuit_breaker_state(self, backend_id: str, state: str, consecutive_failures: int, last_failure_time: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO circuit_breakers (backend_id, state, consecutive_failures, last_failure_time)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(backend_id) DO UPDATE SET
                state=excluded.state,
                consecutive_failures=excluded.consecutive_failures,
                last_failure_time=excluded.last_failure_time
            """, (backend_id, state, consecutive_failures, last_failure_time))
            conn.commit()
