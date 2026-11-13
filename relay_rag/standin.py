from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="The Relay Stand-in Model Service", version="0.1.0")


class SimConfig(BaseModel):
    base_latency_ms: int = Field(default=int(os.getenv("STANDIN_BASE_LATENCY_MS", "350")), ge=0)
    jitter_ms: int = Field(default=int(os.getenv("STANDIN_JITTER_MS", "120")), ge=0)
    per_extra_inflight_ms: int = Field(default=int(os.getenv("STANDIN_PER_EXTRA_INFLIGHT_MS", "180")), ge=0)
    failure_rate: float = Field(default=float(os.getenv("STANDIN_FAILURE_RATE", "0.03")), ge=0, le=1)
    rpm_limit: int = Field(default=int(os.getenv("STANDIN_RPM_LIMIT", "0")), ge=0)
    tpm_limit: int = Field(default=0, ge=0)
    daily_limit: int = Field(default=int(os.getenv("STANDIN_DAILY_LIMIT", "0")), ge=0)
    daily_token_limit: int = Field(default=0, ge=0)
    max_inflight: int = Field(default=int(os.getenv("STANDIN_MAX_INFLIGHT", "0")), ge=0)
    stream_piece_words: int = Field(default=4, ge=1, le=30)


cfg = SimConfig()
lock = asyncio.Lock()
recent_requests: deque[float] = deque()
recent_tokens: deque[tuple[float, int]] = deque()
request_count_today = 0
tokens_today = 0
inflight = 0


def _utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


current_day = _utc_day()


def _headers() -> dict[str, str]:
    return {
        "x-ratelimit-limit-requests": str(cfg.daily_limit if cfg.daily_limit else 999999),
        "x-ratelimit-remaining-requests": str(max(0, cfg.daily_limit - request_count_today) if cfg.daily_limit else 999999),
        "x-ratelimit-limit-tokens": str(cfg.tpm_limit if cfg.tpm_limit else 999999),
        "x-ratelimit-remaining-tokens": "999999",
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "standin", "day": current_day}


@app.get("/admin/state")
async def state() -> dict[str, Any]:
    async with lock:
        return {
            "config": cfg.model_dump(),
            "requests_today": request_count_today,
            "tokens_today_estimated": tokens_today,
            "requests_last_minute": len(recent_requests),
            "tokens_last_minute_estimated": sum(x[1] for x in recent_tokens),
            "inflight": inflight,
            "day_utc": current_day,
        }


@app.post("/admin/config")
async def update_config(new_cfg: SimConfig) -> dict[str, Any]:
    global cfg
    async with lock:
        cfg = new_cfg
    return {"updated": True, "config": cfg.model_dump()}


@app.post("/admin/reset")
async def reset() -> dict[str, Any]:
    global request_count_today, tokens_today, current_day
    async with lock:
        recent_requests.clear()
        recent_tokens.clear()
        request_count_today = 0
        tokens_today = 0
        current_day = _utc_day()
    return {"reset": True}


def _extract_citation(messages: list[dict[str, str]]) -> str:
    # Do not copy the citation-format example from the system prompt. Read only the
    # explicit allowed-citation block supplied by the application.
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if "AVAILABLE CITATIONS" not in content:
            continue
        allowed_block = content.split("AVAILABLE CITATIONS", 1)[1]
        if "RETRIEVED EXCERPTS" in allowed_block:
            allowed_block = allowed_block.split("RETRIEVED EXCERPTS", 1)[0]
        found = re.findall(r"\[([^\],]+),\s*p\.\s*(\d+)\]", allowed_block)
        if found:
            paper, page = found[0]
            return f"[{paper.strip()}, p. {page}]"
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global inflight, request_count_today, tokens_today, current_day
    body = await request.json()
    messages = body.get("messages") or []
    now = time.monotonic()
    tokens_in = max(1, len(str(messages)) // 4)
    citation = _extract_citation(messages)
    text = (
        "Synthetic stand-in response for latency and load testing only. "
        "It is not a factual answer and must not be used to grade retrieval quality. "
        + citation
    ).strip()
    tokens_out = max(1, len(text) // 4)
    tokens_for_request = tokens_in + tokens_out

    async with lock:
        today = _utc_day()
        if today != current_day:
            request_count_today = 0
            tokens_today = 0
            current_day = today
        while recent_requests and now - recent_requests[0] > 60:
            recent_requests.popleft()
        while recent_tokens and now - recent_tokens[0][0] > 60:
            recent_tokens.popleft()
        used_tokens_minute = sum(x[1] for x in recent_tokens)
        if cfg.daily_limit and request_count_today >= cfg.daily_limit:
            raise HTTPException(
                status_code=429,
                detail="stand-in daily request budget exhausted",
                headers={"Retry-After": "3600", **_headers()},
            )
        if cfg.daily_token_limit and tokens_today + tokens_for_request > cfg.daily_token_limit:
            raise HTTPException(
                status_code=429,
                detail="stand-in daily token budget exhausted",
                headers={"Retry-After": "3600", **_headers()},
            )
        if cfg.rpm_limit and len(recent_requests) >= cfg.rpm_limit:
            raise HTTPException(
                status_code=429,
                detail="stand-in requests-per-minute limit reached",
                headers={"Retry-After": "2", **_headers()},
            )
        if cfg.tpm_limit and used_tokens_minute + tokens_for_request > cfg.tpm_limit:
            raise HTTPException(
                status_code=429,
                detail="stand-in tokens-per-minute limit reached",
                headers={"Retry-After": "2", **_headers()},
            )
        if cfg.max_inflight and inflight >= cfg.max_inflight:
            raise HTTPException(
                status_code=429,
                detail="stand-in max in-flight concurrency reached",
                headers={"Retry-After": "1", **_headers()},
            )
        recent_requests.append(now)
        recent_tokens.append((now, tokens_for_request))
        request_count_today += 1
        tokens_today += tokens_for_request
        inflight += 1
        current_inflight = inflight

    stream_owns_inflight = False
    try:
        if random.random() < cfg.failure_rate:
            await asyncio.sleep(min(0.2, cfg.base_latency_ms / 1000))
            raise HTTPException(status_code=503, detail="simulated stand-in transient failure")

        extra_delay = max(0, current_inflight - 1) * cfg.per_extra_inflight_ms
        jitter = random.uniform(0, cfg.jitter_ms)
        await asyncio.sleep((cfg.base_latency_ms + extra_delay + jitter) / 1000.0)

        words = text.split()
        piece_size = cfg.stream_piece_words
        chunks = [" ".join(words[i : i + piece_size]) + " " for i in range(0, len(words), piece_size)]

        async def events():
            global inflight
            try:
                for piece in chunks:
                    payload = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                    yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(0.025)
                usage = {
                    "prompt_tokens": tokens_in,
                    "completion_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                }
                yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                async with lock:
                    inflight = max(0, inflight - 1)

        stream_owns_inflight = True
        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={**_headers(), "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    finally:
        if not stream_owns_inflight:
            async with lock:
                inflight = max(0, inflight - 1)


def main() -> None:
    import uvicorn
    uvicorn.run("relay_rag.standin:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    main()
