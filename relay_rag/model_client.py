from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import settings
from .limiter import DailyBudgetExceeded, MultiRateLimiter

log = logging.getLogger("relay.model_client")


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        headers: dict | None = None,
        attempts: int = 1,
        queue_wait_ms: float = 0.0,
        retry_wait_ms: float = 0.0,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}
        self.attempts = attempts
        self.queue_wait_ms = queue_wait_ms
        self.retry_wait_ms = retry_wait_ms


class ModelTimeoutError(TimeoutError):
    def __init__(self, message: str, attempts: int = 1, queue_wait_ms: float = 0.0, retry_wait_ms: float = 0.0):
        super().__init__(message)
        self.attempts = attempts
        self.queue_wait_ms = queue_wait_ms
        self.retry_wait_ms = retry_wait_ms


@dataclass
class ModelCallResult:
    text: str
    attempts: int
    queue_wait_ms: float
    retry_wait_ms: float
    draft_first_token_ms: float | None
    tokens_in: int
    tokens_out: int
    tokens_source: str
    status: str = "ok"
    provider_queue_ms: float | None = None
    response_headers: dict[str, str] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    # Fallback only. Provider usage is preferred whenever returned in the stream.
    return max(1, (len(text) + 3) // 4)


def _retry_after(headers: dict[str, str]) -> float | None:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


class ChatCompletionClient:
    """OpenAI-compatible streamed chat client for Groq or the local stand-in."""

    def __init__(self, backend: str | None = None):
        self.backend = (backend or settings.backend).lower()
        if self.backend == "groq":
            self.base_url = settings.groq_base_url
            self.api_key = settings.groq_api_key
            self.model = settings.groq_model
        elif self.backend == "standin":
            self.base_url = "http://127.0.0.1:8001/v1"
            self.api_key = "local-standin-key"
            self.model = "standin-chat-model"
        else:
            raise ValueError("LLM_BACKEND must be either 'groq' or 'standin'.")
        self.limiter = MultiRateLimiter()

    @staticmethod
    def _parse_sse_payload(line: str) -> dict[str, Any] | None:
        if not line.startswith("data:"):
            return None
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None

    async def _one_attempt(
        self,
        messages: list[dict[str, str]],
        timeout_seconds: float,
        stream_start: float,
        include_stream_usage: bool = True,
    ) -> tuple[str, dict, dict[str, str], float | None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_completion_tokens": settings.max_output_tokens,
            "stream": True,
        }
        if include_stream_usage:
            payload["stream_options"] = {"include_usage": True}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
        text_parts: list[str] = []
        usage: dict = {}
        response_headers: dict[str, str] = {}
        first_token_ms: float | None = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", headers=headers, json=payload
            ) as response:
                response_headers = {k.lower(): v for k, v in response.headers.items()}
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise ProviderError(
                        f"HTTP {response.status_code}: {body[:1200]}",
                        status_code=response.status_code,
                        headers=response_headers,
                    )
                async for line in response.aiter_lines():
                    data = self._parse_sse_payload(line)
                    if not data:
                        continue
                    if isinstance(data.get("usage"), dict):
                        usage = data["usage"]
                    choices = data.get("choices") or []
                    if choices:
                        delta = (choices[0].get("delta") or {}).get("content")
                        if delta:
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - stream_start) * 1000.0
                            text_parts.append(str(delta))
        return "".join(text_parts), usage, response_headers, first_token_ms

    async def generate(
        self,
        messages: list[dict[str, str]],
        deadline_at: float,
    ) -> ModelCallResult:
        if self.backend == "groq" and not self.api_key:
            raise ProviderError("GROQ_API_KEY is empty. Add it to .env and restart the server.", status_code=401)
        estimated_input = estimate_tokens("\n".join(m.get("content", "") for m in messages))
        estimated_reservation = estimated_input + settings.max_output_tokens
        generate_start = time.perf_counter()
        queue_wait = 0.0
        retry_wait = 0.0
        include_stream_usage = True
        max_attempts = max(1, settings.max_retries + 1)
        last_error: Exception | None = None
        observed_error_headers: dict[str, str] = {}

        for attempt in range(1, max_attempts + 1):
            remaining = deadline_at - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("Overall request deadline expired before model call.")
            try:
                if self.backend == "groq":
                    queue_wait += await self.limiter.acquire(estimated_reservation)
                timeout_s = min(settings.model_call_timeout_seconds, remaining)
                text, usage, headers, draft_first_ms = await asyncio.wait_for(
                    self._one_attempt(messages, timeout_s, generate_start, include_stream_usage), timeout=timeout_s
                )
                if not text.strip():
                    raise ProviderError("Provider returned an empty streamed completion.")
                prompt_tokens = int((usage or {}).get("prompt_tokens") or estimated_input)
                completion_tokens = int((usage or {}).get("completion_tokens") or estimate_tokens(text))
                source = "provider" if usage and usage.get("prompt_tokens") is not None else "estimated"
                return ModelCallResult(
                    text=text,
                    attempts=attempt,
                    queue_wait_ms=queue_wait * 1000.0 + (float(usage["queue_time"]) * 1000.0 if usage.get("queue_time") is not None else 0.0),
                    retry_wait_ms=retry_wait * 1000.0,
                    draft_first_token_ms=draft_first_ms,
                    tokens_in=prompt_tokens,
                    tokens_out=completion_tokens,
                    tokens_source=source,
                    provider_queue_ms=(float(usage["queue_time"]) * 1000.0 if usage.get("queue_time") is not None else None),
                    response_headers={**observed_error_headers, **headers},
                )
            except DailyBudgetExceeded:
                raise
            except asyncio.TimeoutError as exc:
                last_error = exc
                if attempt >= max_attempts:
                    raise ModelTimeoutError(
                        "Model call timed out.", attempts=attempt,
                        queue_wait_ms=queue_wait * 1000.0, retry_wait_ms=retry_wait * 1000.0,
                    ) from exc
                retry_delay = min(settings.retry_max_wait_seconds, 0.5 * (2 ** (attempt - 1)))
            except ProviderError as exc:
                last_error = exc
                observed_error_headers.update(exc.headers)
                if exc.status_code == 400 and include_stream_usage and ("stream_options" in str(exc).lower() or "include_usage" in str(exc).lower()):
                    include_stream_usage = False
                    retry_delay = 0.0
                else:
                    retry_delay = None
                exc.attempts = attempt
                exc.queue_wait_ms = queue_wait * 1000.0
                exc.retry_wait_ms = retry_wait * 1000.0
                if retry_delay is None:
                    if exc.status_code == 429:
                        body_l = str(exc).lower()
                        if "day" in body_l or "daily" in body_l:
                            raise
                        server_wait = _retry_after(exc.headers)
                        if server_wait is None or server_wait > settings.retry_max_wait_seconds:
                            raise
                        retry_delay = server_wait
                    elif exc.status_code is None or exc.status_code >= 500:
                        retry_delay = min(settings.retry_max_wait_seconds, 0.5 * (2 ** (attempt - 1)))
                    else:
                        raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                retry_delay = min(settings.retry_max_wait_seconds, 0.5 * (2 ** (attempt - 1)))

            remaining = deadline_at - time.perf_counter()
            if attempt >= max_attempts or retry_delay > remaining:
                if isinstance(last_error, ProviderError):
                    last_error.attempts = attempt
                    last_error.queue_wait_ms = queue_wait * 1000.0
                    last_error.retry_wait_ms = retry_wait * 1000.0
                    raise last_error
                if isinstance(last_error, (asyncio.TimeoutError, httpx.TimeoutException)):
                    raise ModelTimeoutError(
                        "Model call timed out during retry policy.", attempts=attempt,
                        queue_wait_ms=queue_wait * 1000.0, retry_wait_ms=retry_wait * 1000.0,
                    ) from last_error
                raise ProviderError(f"Transient provider failure: {last_error}") from last_error
            # Small jitter avoids synchronized retries; record the actual wait in the raw metrics.
            retry_delay = min(remaining, retry_delay + random.uniform(0.0, min(0.15, retry_delay * 0.2)))
            await asyncio.sleep(retry_delay)
            retry_wait += retry_delay

        raise ProviderError(f"Model request failed: {last_error}")
