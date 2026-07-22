import sqlite3
import asyncio
import threading
from typing import Dict, Any, Optional
import os
from datetime import datetime, timezone
from gateway.ledger.base_store import BaseLedgerStore

class LedgerStore(BaseLedgerStore):
    """A thread-safe SQLite store for budgets and costs with async methods."""

    def __init__(self, db_path: str = "ledger.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            # Enable write-ahead logging (WAL) mode for concurrency
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    api_key TEXT PRIMARY KEY,
                    daily_limit_usd REAL,
                    monthly_limit_usd REAL,
                    spend_today REAL DEFAULT 0.0,
                    spend_month REAL DEFAULT 0.0,
                    last_reset_date TEXT,
                    last_reset_month TEXT,
                    requests_per_minute INTEGER DEFAULT 60
                )
            """)
            
            # safely migrate existing DB
            try:
                self.conn.execute("ALTER TABLE budgets ADD COLUMN last_reset_date TEXT")
                self.conn.execute("ALTER TABLE budgets ADD COLUMN last_reset_month TEXT")
            except sqlite3.OperationalError:
                pass # columns already exist
            try:
                self.conn.execute("ALTER TABLE budgets ADD COLUMN requests_per_minute INTEGER DEFAULT 60")
            except sqlite3.OperationalError:
                pass
                
            self.conn.execute("""
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
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breakers (
                    backend_id TEXT PRIMARY KEY,
                    state TEXT,
                    consecutive_failures INTEGER,
                    last_failure_time REAL
                )
            """)
            self.conn.commit()

    async def load_budgets_from_config(self, budgets_config: list):
        """Seed initial budgets from config asynchronously."""
        await asyncio.to_thread(self._load_budgets_from_config_sync, budgets_config)

    def _load_budgets_from_config_sync(self, budgets_config: list):
        with self.lock:
            for b in budgets_config:
                rpm = b.get("requests_per_minute", 60)
                self.conn.execute("""
                    INSERT INTO budgets (api_key, daily_limit_usd, monthly_limit_usd, requests_per_minute)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(api_key) DO UPDATE SET
                    daily_limit_usd=excluded.daily_limit_usd,
                    monthly_limit_usd=excluded.monthly_limit_usd,
                    requests_per_minute=excluded.requests_per_minute
                """, (b["api_key"], b["daily_limit_usd"], b["monthly_limit_usd"], rpm))
            self.conn.commit()

    async def get_budget(self, api_key: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_budget_sync, api_key)

    def _get_budget_sync(self, api_key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            self.conn.row_factory = sqlite3.Row
            row = self.conn.execute("SELECT * FROM budgets WHERE api_key = ?", (api_key,)).fetchone()
            if not row:
                return None
                
            budget = dict(row)
            now_utc = datetime.now(timezone.utc)
            today_str = now_utc.strftime('%Y-%m-%d')
            month_str = now_utc.strftime('%Y-%m')
            
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
                self.conn.execute("""
                    UPDATE budgets 
                    SET spend_today = ?, last_reset_date = ?, 
                        spend_month = ?, last_reset_month = ?
                    WHERE api_key = ?
                """, (budget['spend_today'], budget['last_reset_date'],
                      budget['spend_month'], budget['last_reset_month'], api_key))
                self.conn.commit()
                
            return budget

    async def record_request(self, api_key: str, req_id: str, backend: str, model: str, 
                             prompt_tokens: int, comp_tokens: int, cost: float, latency: float):
        await asyncio.to_thread(
            self._record_request_sync,
            api_key, req_id, backend, model, prompt_tokens, comp_tokens, cost, latency
        )

    def _record_request_sync(self, api_key: str, req_id: str, backend: str, model: str, 
                             prompt_tokens: int, comp_tokens: int, cost: float, latency: float):
        with self.lock:
            self.conn.execute("""
                INSERT INTO requests (id, api_key, backend, model, prompt_tokens, completion_tokens, cost_usd, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (req_id, api_key, backend, model, prompt_tokens, comp_tokens, cost, latency))
            
            # Update budget
            self.conn.execute("""
                UPDATE budgets 
                SET spend_today = spend_today + ?, spend_month = spend_month + ?
                WHERE api_key = ?
            """, (cost, cost, api_key))
            self.conn.commit()

    async def get_circuit_breaker_state(self, backend_id: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_circuit_breaker_state_sync, backend_id)

    def _get_circuit_breaker_state_sync(self, backend_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            self.conn.row_factory = sqlite3.Row
            row = self.conn.execute("SELECT * FROM circuit_breakers WHERE backend_id = ?", (backend_id,)).fetchone()
            return dict(row) if row else None

    async def update_circuit_breaker_state(self, backend_id: str, state: str, consecutive_failures: int, last_failure_time: float):
        await asyncio.to_thread(
            self._update_circuit_breaker_state_sync,
            backend_id, state, consecutive_failures, last_failure_time
        )

    def _update_circuit_breaker_state_sync(self, backend_id: str, state: str, consecutive_failures: int, last_failure_time: float):
        with self.lock:
            self.conn.execute("""
                INSERT INTO circuit_breakers (backend_id, state, consecutive_failures, last_failure_time)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(backend_id) DO UPDATE SET
                state=excluded.state,
                consecutive_failures=excluded.consecutive_failures,
                last_failure_time=excluded.last_failure_time
            """, (backend_id, state, consecutive_failures, last_failure_time))
            self.conn.commit()

    async def get_all_budgets(self) -> list:
        return await asyncio.to_thread(self._get_all_budgets_sync)

    def _get_all_budgets_sync(self) -> list:
        with self.lock:
            self.conn.row_factory = sqlite3.Row
            rows = self.conn.execute("SELECT api_key, daily_limit_usd, spend_today, monthly_limit_usd, spend_month FROM budgets").fetchall()
            return [dict(r) for r in rows]

    async def get_all_requests(self, limit: int = 50) -> list:
        return await asyncio.to_thread(self._get_all_requests_sync, limit)

    def _get_all_requests_sync(self, limit: int) -> list:
        with self.lock:
            self.conn.row_factory = sqlite3.Row
            rows = self.conn.execute(
                "SELECT id, api_key, backend, model, cost_usd, latency_ms, timestamp FROM requests ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    async def get_all_circuit_breakers(self) -> list:
        return await asyncio.to_thread(self._get_all_circuit_breakers_sync)

    def _get_all_circuit_breakers_sync(self) -> list:
        with self.lock:
            self.conn.row_factory = sqlite3.Row
            rows = self.conn.execute("SELECT backend_id, state, consecutive_failures, last_failure_time FROM circuit_breakers").fetchall()
            return [dict(r) for r in rows]

    async def update_budget_limits(self, api_key: str, daily_limit_usd: float, monthly_limit_usd: float, requests_per_minute: Optional[int] = None) -> bool:
        return await asyncio.to_thread(self._update_budget_limits_sync, api_key, daily_limit_usd, monthly_limit_usd, requests_per_minute)

    def _update_budget_limits_sync(self, api_key: str, daily_limit_usd: float, monthly_limit_usd: float, requests_per_minute: Optional[int] = None) -> bool:
        with self.lock:
            # Check if key exists
            row = self.conn.execute("SELECT api_key, requests_per_minute FROM budgets WHERE api_key = ?", (api_key,)).fetchone()
            rpm = requests_per_minute if requests_per_minute is not None else (row[1] if row else 60)
            if row:
                self.conn.execute("""
                    UPDATE budgets 
                    SET daily_limit_usd = ?, monthly_limit_usd = ?, requests_per_minute = ?
                    WHERE api_key = ?
                """, (daily_limit_usd, monthly_limit_usd, rpm, api_key))
                self.conn.commit()
                return True
            else:
                self.conn.execute("""
                    INSERT INTO budgets (api_key, daily_limit_usd, monthly_limit_usd, requests_per_minute, spend_today, spend_month)
                    VALUES (?, ?, ?, ?, 0.0, 0.0)
                """, (api_key, daily_limit_usd, monthly_limit_usd, rpm))
                self.conn.commit()
                return False

    def get_all_api_keys_sync(self) -> list:
        with self.lock:
            rows = self.conn.execute("SELECT api_key FROM budgets").fetchall()
            return [r[0] for r in rows]

    def get_all_api_keys_and_limits_sync(self) -> list:
        with self.lock:
            rows = self.conn.execute("SELECT api_key, requests_per_minute FROM budgets").fetchall()
            return [{"api_key": r[0], "requests_per_minute": r[1]} for r in rows]

    def close(self):
        """Close the SQLite connection."""
        with self.lock:
            self.conn.close()

