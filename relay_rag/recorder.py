from __future__ import annotations

import asyncio
import csv
from pathlib import Path

from .config import settings

REQUIRED_COLUMNS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
    "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
]
EXTRA_COLUMNS = [
    "queue_wait_ms", "work_ms", "retry_wait_ms", "draft_first_token_ms", "top_score",
    "citation_attempts", "retrieval_tool_calls", "citation_valid", "tokens_source", "error_kind", "provider_error",
    "rate_limit_limit_requests_rpd", "rate_limit_remaining_requests_rpd",
    "rate_limit_limit_tokens_tpm", "rate_limit_remaining_tokens_tpm", "retry_after", "provider_queue_ms",
    "query_rewrite_ms", "retrieval_ms", "generation_ms", "citation_check_ms",
    "client_queue_depth", "client_inflight", "cost_usd",
    "citation_validation_reason", "citation_invalid",
]
ALL_COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS


class ResultRecorder:
    def __init__(self, path: Path | None = None):
        self.path = path or settings.results_csv
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=ALL_COLUMNS).writeheader()

    async def append(self, row: dict) -> None:
        async with self.lock:
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow({key: row.get(key, "") for key in ALL_COLUMNS})
