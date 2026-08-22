import asyncio
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from gateway.ledger.base_store import BaseLedgerStore


class LedgerStore(BaseLedgerStore):
    """A thread-safe SQLite store for budgets and costs with thread-isolated connections."""

    def __init__(self, db_path: str = "ledger.db", timeout: float = 30.0):
        self.db_path = db_path
        self.timeout = timeout
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._is_memory = db_path == ":memory:" or "mode=memory" in db_path
        self._memory_conn: Optional[sqlite3.Connection] = None
        self._all_conns: list = []

        if self._is_memory:
            # For in-memory DB (used in tests), maintain a shared connection handle
            self._memory_conn = sqlite3.connect(
                self.db_path, timeout=self.timeout, check_same_thread=False
            )
            self._memory_conn.row_factory = sqlite3.Row
            self._all_conns.append(self._memory_conn)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-isolated SQLite connection configured for concurrent operation."""
        if self._is_memory and self._memory_conn is not None:
            return self._memory_conn

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.db_path, timeout=self.timeout, check_same_thread=False
            )
            conn.isolation_level = (
                None  # Autocommit mode: prevent dangling transaction locks
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000").fetchall()
            self._local.conn = conn
            with self._write_lock:
                self._all_conns.append(conn)
        return conn

    def _init_db(self):
        with self._write_lock:
            conn = self._get_connection()
            if not self._is_memory:
                conn.execute("PRAGMA journal_mode=WAL").fetchall()
            conn.execute("PRAGMA synchronous=NORMAL").fetchall()
            conn.execute("PRAGMA busy_timeout=30000").fetchall()
            conn.execute("""
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
                conn.execute("ALTER TABLE budgets ADD COLUMN last_reset_date TEXT")
                conn.execute("ALTER TABLE budgets ADD COLUMN last_reset_month TEXT")
            except sqlite3.OperationalError:
                pass  # columns already exist
            try:
                conn.execute(
                    "ALTER TABLE budgets ADD COLUMN requests_per_minute INTEGER DEFAULT 60"
                )
            except sqlite3.OperationalError:
                pass

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
            # Sliding-window rate-limit log: one row per admitted request.
            # Rows older than 60 s are pruned on each check so the table
            # stays small. Using the shared DB makes limits survive restarts
            # and are enforced consistently across all workers/threads.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key   TEXT    NOT NULL,
                    ts        REAL    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rll_key_ts ON rate_limit_log (api_key, ts)"
            )
            # Quality labels. Routing decisions in this gateway are otherwise
            # unfalsifiable: nothing records whether the model that answered
            # actually produced a good answer, so no routing change can be
            # validated and a learned router has nothing to train on.
            # Capturing a rating per request is the smallest thing that makes
            # that measurable, and accumulates the labelled data such a router
            # would need. See F13 in docs/AI_ML_FLAWS.md.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT    NOT NULL,
                    api_key    TEXT,
                    backend    TEXT,
                    model      TEXT,
                    rating     INTEGER NOT NULL,
                    comment    TEXT,
                    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_request "
                "ON feedback (request_id)"
            )
            conn.commit()

    async def load_budgets_from_config(self, budgets_config: list):
        """Seed initial budgets from config asynchronously."""
        await asyncio.to_thread(self._load_budgets_from_config_sync, budgets_config)

    def _load_budgets_from_config_sync(self, budgets_config: list):
        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        month_str = now_utc.strftime("%Y-%m")
        with self._write_lock:
            conn = self._get_connection()
            for b in budgets_config:
                rpm = b.get("requests_per_minute", 60)
                conn.execute(
                    """
                    INSERT INTO budgets (api_key, daily_limit_usd, monthly_limit_usd, requests_per_minute, last_reset_date, last_reset_month)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(api_key) DO UPDATE SET
                    daily_limit_usd=excluded.daily_limit_usd,
                    monthly_limit_usd=excluded.monthly_limit_usd,
                    requests_per_minute=excluded.requests_per_minute
                """,
                    (
                        b["api_key"],
                        b["daily_limit_usd"],
                        b["monthly_limit_usd"],
                        rpm,
                        today_str,
                        month_str,
                    ),
                )
            conn.commit()

    async def get_budget(self, api_key: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_budget_sync, api_key)

    def _get_budget_sync(self, api_key: str) -> Optional[Dict[str, Any]]:
        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        month_str = now_utc.strftime("%Y-%m")

        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM budgets WHERE api_key = ?", (api_key,)
        ).fetchone()
        if not row:
            return None

        budget = dict(row)

        needs_daily_reset = budget.get("last_reset_date") != today_str
        needs_monthly_reset = budget.get("last_reset_month") != month_str

        if needs_daily_reset or needs_monthly_reset:
            with self._write_lock:
                w_conn = self._get_connection()
                w_row = w_conn.execute(
                    "SELECT * FROM budgets WHERE api_key = ?", (api_key,)
                ).fetchone()
                if w_row:
                    budget = dict(w_row)
                    if budget.get("last_reset_date") != today_str:
                        budget["spend_today"] = 0.0
                        budget["last_reset_date"] = today_str
                    if budget.get("last_reset_month") != month_str:
                        budget["spend_month"] = 0.0
                        budget["last_reset_month"] = month_str
                    w_conn.execute(
                        """
                        UPDATE budgets 
                        SET spend_today = ?, last_reset_date = ?, 
                            spend_month = ?, last_reset_month = ?
                        WHERE api_key = ?
                    """,
                        (
                            budget["spend_today"],
                            budget["last_reset_date"],
                            budget["spend_month"],
                            budget["last_reset_month"],
                            api_key,
                        ),
                    )
                    w_conn.commit()

        return budget

    async def record_request(
        self,
        api_key: str,
        req_id: str,
        backend: str,
        model: str,
        prompt_tokens: int,
        comp_tokens: int,
        cost: float,
        latency: float,
        reserved_cost: float = 0.0,
    ):
        await asyncio.to_thread(
            self._record_request_sync,
            api_key,
            req_id,
            backend,
            model,
            prompt_tokens,
            comp_tokens,
            cost,
            latency,
            reserved_cost,
        )

    def _record_request_sync(
        self,
        api_key: str,
        req_id: str,
        backend: str,
        model: str,
        prompt_tokens: int,
        comp_tokens: int,
        cost: float,
        latency: float,
        reserved_cost: float = 0.0,
    ):
        with self._write_lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO requests (id, api_key, backend, model, prompt_tokens, completion_tokens, cost_usd, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    req_id,
                    api_key,
                    backend,
                    model,
                    prompt_tokens,
                    comp_tokens,
                    cost,
                    latency,
                ),
            )

            # The reservation already pre-incremented spend by reserved_cost.
            # Apply the delta between actual cost and the reservation.
            delta = cost - reserved_cost
            conn.execute(
                """
                UPDATE budgets 
                SET spend_today = spend_today + ?, spend_month = spend_month + ?
                WHERE api_key = ?
            """,
                (delta, delta, api_key),
            )
            conn.commit()

    async def get_circuit_breaker_state(
        self, backend_id: str
    ) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_circuit_breaker_state_sync, backend_id)

    def _get_circuit_breaker_state_sync(
        self, backend_id: str
    ) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM circuit_breakers WHERE backend_id = ?", (backend_id,)
        ).fetchone()
        return dict(row) if row else None

    async def update_circuit_breaker_state(
        self,
        backend_id: str,
        state: str,
        consecutive_failures: int,
        last_failure_time: float,
    ):
        await asyncio.to_thread(
            self._update_circuit_breaker_state_sync,
            backend_id,
            state,
            consecutive_failures,
            last_failure_time,
        )

    def _update_circuit_breaker_state_sync(
        self,
        backend_id: str,
        state: str,
        consecutive_failures: int,
        last_failure_time: float,
    ):
        with self._write_lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO circuit_breakers (backend_id, state, consecutive_failures, last_failure_time)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(backend_id) DO UPDATE SET
                state=excluded.state,
                consecutive_failures=excluded.consecutive_failures,
                last_failure_time=excluded.last_failure_time
            """,
                (backend_id, state, consecutive_failures, last_failure_time),
            )
            conn.commit()

    async def get_all_budgets(self) -> list:
        return await asyncio.to_thread(self._get_all_budgets_sync)

    def _get_all_budgets_sync(self) -> list:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT api_key, daily_limit_usd, spend_today, monthly_limit_usd, spend_month FROM budgets"
        ).fetchall()
        return [dict(r) for r in rows]

    async def get_all_requests(self, limit: int = 50) -> list:
        return await asyncio.to_thread(self._get_all_requests_sync, limit)

    def _get_all_requests_sync(self, limit: int) -> list:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT id, api_key, backend, model, cost_usd, latency_ms, timestamp FROM requests ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    async def record_feedback(
        self,
        request_id: str,
        rating: int,
        api_key: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        """Store a quality label for a request.

        Backend and model are resolved from the ledger rather than trusted
        from the caller, so a label is always attributed to whatever actually
        answered — which is the point of collecting it.
        """
        await asyncio.to_thread(
            self._record_feedback_sync, request_id, rating, api_key, comment
        )

    def _record_feedback_sync(
        self,
        request_id: str,
        rating: int,
        api_key: Optional[str],
        comment: Optional[str],
    ) -> None:
        with self._write_lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT api_key, backend, model FROM requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            conn.execute(
                "INSERT INTO feedback (request_id, api_key, backend, model, "
                "rating, comment) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    (row["api_key"] if row else None) or api_key,
                    row["backend"] if row else None,
                    row["model"] if row else None,
                    rating,
                    comment,
                ),
            )
            conn.commit()

    async def get_feedback_summary(self) -> list:
        """Per-backend rating counts — the beginnings of a quality signal."""
        return await asyncio.to_thread(self._get_feedback_summary_sync)

    def _get_feedback_summary_sync(self) -> list:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT backend, model, COUNT(*) AS total, "
            "SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) AS positive, "
            "SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) AS negative "
            "FROM feedback GROUP BY backend, model ORDER BY total DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    async def get_all_circuit_breakers(self) -> list:
        return await asyncio.to_thread(self._get_all_circuit_breakers_sync)

    def _get_all_circuit_breakers_sync(self) -> list:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT backend_id, state, consecutive_failures, last_failure_time FROM circuit_breakers"
        ).fetchall()
        return [dict(r) for r in rows]

    async def update_budget_limits(
        self,
        api_key: str,
        daily_limit_usd: float,
        monthly_limit_usd: float,
        requests_per_minute: Optional[int] = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._update_budget_limits_sync,
            api_key,
            daily_limit_usd,
            monthly_limit_usd,
            requests_per_minute,
        )

    def _update_budget_limits_sync(
        self,
        api_key: str,
        daily_limit_usd: float,
        monthly_limit_usd: float,
        requests_per_minute: Optional[int] = None,
    ) -> bool:
        with self._write_lock:
            conn = self._get_connection()
            # Check if key exists
            row = conn.execute(
                "SELECT api_key, requests_per_minute FROM budgets WHERE api_key = ?",
                (api_key,),
            ).fetchone()
            rpm = (
                requests_per_minute
                if requests_per_minute is not None
                else (row[1] if row else 60)
            )
            if row:
                conn.execute(
                    """
                    UPDATE budgets 
                    SET daily_limit_usd = ?, monthly_limit_usd = ?, requests_per_minute = ?
                    WHERE api_key = ?
                """,
                    (daily_limit_usd, monthly_limit_usd, rpm, api_key),
                )
                conn.commit()
                return True
            else:
                conn.execute(
                    """
                    INSERT INTO budgets (api_key, daily_limit_usd, monthly_limit_usd, requests_per_minute, spend_today, spend_month)
                    VALUES (?, ?, ?, ?, 0.0, 0.0)
                """,
                    (api_key, daily_limit_usd, monthly_limit_usd, rpm),
                )
                conn.commit()
                return False

    async def check_and_reserve_budget(
        self, api_key: str, estimated_cost: float
    ) -> Dict[str, Any]:
        """Atomically check limits and pre-increment spend by estimated_cost."""
        return await asyncio.to_thread(
            self._check_and_reserve_budget_sync, api_key, estimated_cost
        )

    def _check_and_reserve_budget_sync(
        self, api_key: str, estimated_cost: float
    ) -> Dict[str, Any]:
        """Single-transaction atomic check + reserve under the write lock."""
        from gateway.policy.budget import BudgetExceededException

        with self._write_lock:
            conn = self._get_connection()

            # ── 1. Handle period resets inside the same lock / transaction ──
            row = conn.execute(
                "SELECT * FROM budgets WHERE api_key = ?", (api_key,)
            ).fetchone()

            if not row:
                raise ValueError(f"No budget found for API key: {api_key}")

            budget = dict(row)
            now_utc = datetime.now(timezone.utc)
            today_str = now_utc.strftime("%Y-%m-%d")
            month_str = now_utc.strftime("%Y-%m")

            needs_reset = False
            if budget.get("last_reset_date") != today_str:
                budget["spend_today"] = 0.0
                budget["last_reset_date"] = today_str
                needs_reset = True
            if budget.get("last_reset_month") != month_str:
                budget["spend_month"] = 0.0
                budget["last_reset_month"] = month_str
                needs_reset = True

            if needs_reset:
                conn.execute(
                    """
                    UPDATE budgets
                    SET spend_today = ?, last_reset_date = ?,
                        spend_month = ?, last_reset_month = ?
                    WHERE api_key = ?
                """,
                    (
                        budget["spend_today"],
                        budget["last_reset_date"],
                        budget["spend_month"],
                        budget["last_reset_month"],
                        api_key,
                    ),
                )

            # ── 2. Check limits ──────────────────────────────────────────────
            projected_daily = budget["spend_today"] + estimated_cost
            if (
                budget["daily_limit_usd"]
                and projected_daily > budget["daily_limit_usd"]
            ):
                conn.commit()  # commit any reset
                raise BudgetExceededException(
                    f"Daily budget exceeded. "
                    f"Limit: ${budget['daily_limit_usd']}, "
                    f"Projected: ${projected_daily:.6f}"
                )

            projected_monthly = budget["spend_month"] + estimated_cost
            if (
                budget["monthly_limit_usd"]
                and projected_monthly > budget["monthly_limit_usd"]
            ):
                conn.commit()
                raise BudgetExceededException(
                    f"Monthly budget exceeded. "
                    f"Limit: ${budget['monthly_limit_usd']}, "
                    f"Projected: ${projected_monthly:.6f}"
                )

            # ── 3. Reserve – atomically increment spend ──────────────────────
            conn.execute(
                """
                UPDATE budgets
                SET spend_today = spend_today + ?,
                    spend_month = spend_month + ?
                WHERE api_key = ?
            """,
                (estimated_cost, estimated_cost, api_key),
            )
            conn.commit()

            budget["spend_today"] = projected_daily
            budget["spend_month"] = projected_monthly
            return budget

    def get_all_api_keys_sync(self) -> list:
        conn = self._get_connection()
        rows = conn.execute("SELECT api_key FROM budgets").fetchall()
        return [r[0] for r in rows]

    def get_all_api_keys_and_limits_sync(self) -> list:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT api_key, requests_per_minute FROM budgets"
        ).fetchall()
        return [{"api_key": r[0], "requests_per_minute": r[1]} for r in rows]

    async def check_rate_limit(self, api_key: str) -> None:
        """Atomically enforce the per-minute sliding-window rate limit.

        Raises HTTPException(429) if the key has already been admitted
        *requests_per_minute* times in the last 60 seconds.

        Persists to the shared SQLite DB so the counter survives process
        restarts and is consistent across threads/workers using the same file.
        """
        await asyncio.to_thread(self._check_rate_limit_sync, api_key)

    def _check_rate_limit_sync(self, api_key: str) -> None:
        import time

        from fastapi import HTTPException

        now = time.time()
        window_start = now - 60.0

        with self._write_lock:
            conn = self._get_connection()
            # 1. Prune stale rows for this key
            conn.execute(
                "DELETE FROM rate_limit_log WHERE api_key = ? AND ts < ?",
                (api_key, window_start),
            )

            # 2. Count requests in the current window
            count = conn.execute(
                "SELECT COUNT(*) FROM rate_limit_log WHERE api_key = ? AND ts >= ?",
                (api_key, window_start),
            ).fetchone()[0]

            # 3. Fetch the configured limit for this key (default 60)
            row = conn.execute(
                "SELECT requests_per_minute FROM budgets WHERE api_key = ?",
                (api_key,),
            ).fetchone()
            limit = row[0] if row and row[0] is not None else 60

            if count >= limit:
                conn.commit()  # persist the prune
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {limit} requests per minute.",
                )

            # 4. Admit the request – record it
            conn.execute(
                "INSERT INTO rate_limit_log (api_key, ts) VALUES (?, ?)",
                (api_key, now),
            )
            conn.commit()

    def close(self):
        """Close all open SQLite connections."""
        with self._write_lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
            if self._memory_conn is not None:
                try:
                    self._memory_conn.close()
                except Exception:
                    pass
                self._memory_conn = None
