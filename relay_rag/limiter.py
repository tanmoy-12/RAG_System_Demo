from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, settings


class DailyBudgetExceeded(RuntimeError):
    pass


class TokenBucket:
    def __init__(self, rate_per_minute: int, capacity: int):
        self.enabled = rate_per_minute > 0
        self.rate_per_second = rate_per_minute / 60.0 if self.enabled else 0.0
        self.capacity = float(max(1, capacity)) if self.enabled else float("inf")
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> float:
        if not self.enabled:
            return 0.0
        if amount > self.capacity:
            raise ValueError(
                f"Request needs {amount:.0f} tokens, larger than the configured bucket capacity "
                f"({self.capacity:.0f}); lower context/output or verify TPM configuration."
            )
        waited = 0.0
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.updated
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
                self.updated = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return waited
                sleep_for = (amount - self.tokens) / self.rate_per_second
            await asyncio.sleep(sleep_for)
            waited += sleep_for


class DailyBudget:
    """Persistent local guardrail. Limits are only enforced when account values are configured."""

    def __init__(self) -> None:
        self.path = ROOT / "data" / "runtime_usage.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS daily_usage (day TEXT PRIMARY KEY, requests INTEGER NOT NULL, tokens INTEGER NOT NULL)"
            )

    async def reserve(self, estimated_tokens: int) -> None:
        if settings.groq_rpd <= 0 and settings.groq_tpd <= 0:
            return
        day = datetime.now(timezone.utc).date().isoformat()
        async with self.lock:
            with sqlite3.connect(self.path) as conn:
                row = conn.execute("SELECT requests, tokens FROM daily_usage WHERE day=?", (day,)).fetchone()
                requests, tokens = row if row else (0, 0)
                if settings.groq_rpd > 0 and requests + 1 > settings.groq_rpd:
                    raise DailyBudgetExceeded("Configured Groq requests-per-day budget is exhausted.")
                if settings.groq_tpd > 0 and tokens + estimated_tokens > settings.groq_tpd:
                    raise DailyBudgetExceeded("Configured Groq tokens-per-day budget would be exceeded.")
                conn.execute(
                    "INSERT INTO daily_usage(day, requests, tokens) VALUES(?,?,?) "
                    "ON CONFLICT(day) DO UPDATE SET requests=excluded.requests, tokens=excluded.tokens",
                    (day, requests + 1, tokens + estimated_tokens),
                )


class MultiRateLimiter:
    def __init__(self) -> None:
        self.requests = TokenBucket(settings.groq_rpm, settings.groq_rpm)
        self.tokens = TokenBucket(settings.groq_tpm, settings.groq_tpm)
        self.daily = DailyBudget()

    async def acquire(self, estimated_tokens: int) -> float:
        await self.daily.reserve(estimated_tokens)
        request_wait = await self.requests.acquire(1)
        token_wait = await self.tokens.acquire(max(1, estimated_tokens))
        return request_wait + token_wait
