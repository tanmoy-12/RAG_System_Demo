from __future__ import annotations

import json
import logging
import re

import httpx

from .config import settings

log = logging.getLogger("relay.query_rewriter")


class OllamaQueryRewriter:
    """Small local model only rewrites the question; it never decides factual answers."""

    async def rewrite(self, question: str) -> tuple[str, str]:
        prompt = (
            "Rewrite the user's question as one concise, standalone search query for retrieving passages "
            "from a fixed library of three research papers. Preserve technical terms, entities, and intent. "
            "Do not answer the question. Do not add facts. Return only JSON: "
            '{"standalone_query":"..."}.\n\nQuestion: ' + question
        )
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": "You rewrite retrieval queries. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 2048},
        }
        try:
            timeout = httpx.Timeout(settings.ollama_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
                response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            parsed = json.loads(content)
            rewritten = str(parsed.get("standalone_query", "")).strip()
            if not rewritten:
                raise ValueError("Ollama returned an empty standalone_query")
            rewritten = re.sub(r"\s+", " ", rewritten)[:1000]
            return rewritten, "ollama"
        except Exception as exc:  # noqa: BLE001 - fallback is intentional for availability
            log.warning("Query rewrite unavailable; using original query: %s", exc)
            return question, "fallback_original"
