from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone

from .citations import validate_citations
from .config import RETRIEVAL_SETTINGS, settings
from .limiter import DailyBudgetExceeded
from .model_client import ChatCompletionClient, ModelTimeoutError, ProviderError
from .query_rewriter import OllamaQueryRewriter
from .recorder import ResultRecorder
from .retriever import RetrievedChunk, RetrieverRegistry

log = logging.getLogger("relay.pipeline")

async def retrieve_papers_tool(registry: RetrieverRegistry, query_variants: list[str], setting: str):
    """Local retrieval tool: vector-searches the indexed paper chunks and returns provenance metadata."""
    return await asyncio.to_thread(registry.search, query_variants, setting)


SYSTEM_PROMPT = r"""You are The Relay, a research-paper question answering assistant.
You MUST answer only from the retrieved excerpts supplied in the user message. Do not use outside knowledge or memory.
If the excerpts do not establish the answer, say explicitly that the provided papers/excerpts do not state it or that the evidence is insufficient. Do not guess.
Organize the answer into clear logical sections. Include at least one or more citations copied exactly from the AVAILABLE CITATIONS list in every substantive section, in this format: [Paper citation name, p. 12]. Add citations to individual factual paragraphs or bullets where practical, especially when claims come from different pages.
For a displayed equation written with \[ ... \], $$ ... $$, or a LaTeX equation environment, cite the equation immediately after the closing delimiter when practical. A displayed equation may share the citation of its surrounding logical section, but the section must contain a valid citation. Inline equations inherit the citation of their containing prose paragraph.
Use only citations listed as available. Do not invent page numbers, paper names, quotations, or results.
Distinguish clearly between what the authors report and any cautious inference. Prefer a concise, direct answer.
"""


class RAGPipeline:
    def __init__(self, backend: str | None = None):
        self.retrievers = RetrieverRegistry()
        self.rewriter = OllamaQueryRewriter()
        self.llm = ChatCompletionClient(backend=backend)
        self.recorder = ResultRecorder()

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
        blocks = []
        allowed = []
        seen = set()
        for i, chunk in enumerate(chunks, start=1):
            citation = chunk.citation
            if citation not in seen:
                allowed.append(citation)
                seen.add(citation)
            blocks.append(
                f"[RETRIEVED CHUNK {i}]\n"
                f"chunk_id: {chunk.chunk_id}\n"
                f"paper: {chunk.paper}\n"
                f"page: {chunk.page}\n"
                f"citation: {citation}\n"
                f"semantic_similarity: {chunk.semantic_score:.4f}\n"
                f"text: {chunk.text}"
            )
        return "\n\n".join(blocks), allowed

    @staticmethod
    def _refusal(message: str) -> str:
        return message

    async def answer(
        self,
        question: str,
        retrieval_setting: str | None = None,
        scenario: str | None = None,
        client_queue_depth: int = 0,
        client_inflight: int = 0,
    ) -> dict:
        """Enforce an end-to-end deadline, including local rewrite and retrieval stages."""
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(
                self._answer_impl(question, retrieval_setting, scenario, client_queue_depth, client_inflight),
                timeout=settings.request_deadline_seconds,
            )
        except asyncio.TimeoutError:
            request_id = str(uuid.uuid4())
            start_dt = datetime.now(timezone.utc).isoformat()
            total_ms = (time.perf_counter() - started) * 1000.0
            answer_text = "The request exceeded its overall deadline before a verified answer was ready. Please retry."
            result = {
                "request_id": request_id, "question": question, "answer": answer_text,
                "status": "timeout", "source": "stand-in" if self.llm.backend == "standin" else "real",
                "scenario": scenario or "manual", "start_time": start_dt, "first_token_ms": "",
                "draft_first_token_ms": "", "total_ms": round(total_ms, 2), "retries": 0,
                "tokens_in": 0, "tokens_out": 0, "tokens_source": "none",
                "retrieval_setting": retrieval_setting or settings.default_retrieval_setting,
                "top_score": "", "retrieved_chunks": [], "citation_attempts": 0,
                "retrieval_tool_calls": 0, "citation_valid": "", "rewrite_source": "not_measured",
                "queue_wait_ms": 0.0, "provider_queue_ms": "", "work_ms": round(total_ms, 2),
                "query_rewrite_ms": "", "retrieval_ms": "", "generation_ms": "",
                "citation_check_ms": "", "retry_wait_ms": 0.0, "error_kind": "overall_deadline",
                "provider_error": "", "rate_limit_limit_requests_rpd": "", "rate_limit_remaining_requests_rpd": "",
                "rate_limit_limit_tokens_tpm": "", "rate_limit_remaining_tokens_tpm": "", "retry_after": "",
                "client_queue_depth": client_queue_depth, "client_inflight": client_inflight, "cost_usd": 0.0,
            }
            await self.recorder.append({
                "request_id": request_id, "scenario": scenario or "manual", "source": result["source"],
                "start_time": start_dt, "first_token_ms": "", "total_ms": result["total_ms"],
                "status": "timeout", "retries": 0, "tokens_in": 0, "tokens_out": 0,
                "retrieval_setting": result["retrieval_setting"], "answer_grade": "",
                "queue_wait_ms": 0.0, "provider_queue_ms": "", "work_ms": result["work_ms"],
                "retry_wait_ms": 0.0, "draft_first_token_ms": "", "top_score": "",
                "citation_attempts": 0, "retrieval_tool_calls": 0, "citation_valid": "",
                "tokens_source": "none", "error_kind": "overall_deadline",
                "provider_error": "", "rate_limit_limit_requests_rpd": "", "rate_limit_remaining_requests_rpd": "",
                "rate_limit_limit_tokens_tpm": "", "rate_limit_remaining_tokens_tpm": "", "retry_after": "",
                "client_queue_depth": client_queue_depth, "client_inflight": client_inflight, "cost_usd": 0.0,
            })
            return result

    async def _answer_impl(
        self,
        question: str,
        retrieval_setting: str | None = None,
        scenario: str | None = None,
        client_queue_depth: int = 0,
        client_inflight: int = 0,
    ) -> dict:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")
        if len(question) > 4000:
            raise ValueError("Question is too long (maximum 4,000 characters).")
        request_id = str(uuid.uuid4())
        start_dt = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        deadline_at = start + settings.request_deadline_seconds
        retrieval_setting = retrieval_setting or settings.default_retrieval_setting
        if retrieval_setting not in RETRIEVAL_SETTINGS:
            raise ValueError(f"Unknown retrieval_setting '{retrieval_setting}'.")

        queue_wait_ms = 0.0
        provider_queue_ms = 0.0
        retry_wait_ms = 0.0
        query_rewrite_ms = 0.0
        retrieval_ms = 0.0
        generation_ms = 0.0
        citation_check_ms = 0.0
        draft_first_token_ms: float | None = None
        tokens_in = 0
        tokens_out = 0
        tokens_source_set: set[str] = set()
        retries = 0
        citation_attempts = 0
        citation_valid: bool | None = None
        citation_validation_reason = "not_run"
        citation_invalid: list[str] = []
        citation_uncited_blocks: list[str] = []
        citation_found: list[str] = []
        top_score: float | None = None
        error_kind = ""
        response_headers: dict[str, str] = {}
        provider_error = ""
        status = "ok"
        visible_first_token_ms: float | None = None
        answer_text = ""
        retrieved: list[RetrievedChunk] = []

        try:
            stage_start = time.perf_counter()
            rewritten, rewrite_source = await self.rewriter.rewrite(question)
            query_rewrite_ms = (time.perf_counter() - stage_start) * 1000.0
            query_variants = [question]
            if rewritten.strip() and rewritten.strip().casefold() != question.strip().casefold():
                query_variants.append(rewritten)

            stage_start = time.perf_counter()
            # Explicit tool invocation is deterministic and always precedes any hosted generation call.
            retrieved, top_score = await retrieve_papers_tool(
                self.retrievers, query_variants, retrieval_setting
            )
            retrieval_ms = (time.perf_counter() - stage_start) * 1000.0
            if top_score < settings.min_retrieval_score:
                answer_text = self._refusal(
                    "I cannot answer this reliably from the three supplied papers: the strongest retrieved passage "
                    f"was below the configured relevance threshold ({top_score:.3f} < "
                    f"{settings.min_retrieval_score:.3f}). No hosted model was called."
                )
            else:
                context, allowed_citations = self._build_context(retrieved)
                user_content = (
                    f"QUESTION:\n{question}\n\n"
                    f"REWRITTEN SEARCH QUERY (for retrieval only):\n{rewritten}\n\n"
                    f"AVAILABLE CITATIONS (you may use only these exact paper/page pairs):\n"
                    + "\n".join(allowed_citations)
                    + "\n\nRETRIEVED EXCERPTS:\n"
                    + context
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                accepted = False
                for attempt_index in range(2):  # one initial draft + at most one citation repair
                    citation_attempts += 1
                    if attempt_index == 1:
                        retries += 1  # citation-repair generation is one logical retry
                    model_pre_wait_ms = (time.perf_counter() - start) * 1000.0
                    generation_start = time.perf_counter()
                    try:
                        model_result = await self.llm.generate(messages, deadline_at)
                    finally:
                        generation_ms += (time.perf_counter() - generation_start) * 1000.0
                    queue_wait_ms += model_result.queue_wait_ms
                    if model_result.provider_queue_ms is not None:
                        provider_queue_ms += model_result.provider_queue_ms
                    retry_wait_ms += model_result.retry_wait_ms
                    retries += max(0, model_result.attempts - 1)
                    tokens_in += model_result.tokens_in
                    tokens_out += model_result.tokens_out
                    tokens_source_set.add(model_result.tokens_source)
                    response_headers.update(model_result.response_headers)
                    if model_result.draft_first_token_ms is not None and draft_first_token_ms is None:
                        draft_first_token_ms = model_pre_wait_ms + model_result.draft_first_token_ms
                    check_start = time.perf_counter()
                    check = validate_citations(model_result.text, retrieved)
                    citation_check_ms += (time.perf_counter() - check_start) * 1000.0
                    citation_validation_reason = check.reason
                    citation_invalid = check.invalid
                    citation_uncited_blocks = check.uncited_blocks
                    citation_found = check.found
                    if check.valid:
                        answer_text = model_result.text.strip()
                        citation_valid = True
                        accepted = True
                        break
                    citation_valid = False
                    if attempt_index == 0:
                        messages = messages + [
                            {"role": "assistant", "content": model_result.text},
                            {
                                "role": "user",
                                "content": (
                                    "Your previous draft failed citation validation. Repair it once without adding facts. "
                                    f"Validation issue: {check.reason} "
                                    f"Invalid citations: {check.invalid}. "
                                    f"Uncited factual blocks: {check.uncited_blocks}. "
                                    "Use ONLY these exact paper/page citations, copied character-for-character:\n"
                                    + "\n".join(allowed_citations)
                                    + "\nOrganize the answer into logical sections and ensure every substantive section contains at least one valid citation. "
                                    "Add citations to individual factual paragraphs and bullets where practical. "
                                    "A displayed equation may share its section citation, but keep the citation close to the equation when practical. "
                                    "Keep ordinary wrapped lines together as paragraphs. Do not cite a page unless it is in the list. "
                                    "If the retrieved excerpts do not support the answer, respond only with a clear refusal."
                                ),
                            },
                        ]
                if not accepted:
                    status = "error"
                    error_kind = "citation_validation_failed"
                    answer_text = self._refusal(
                        "I cannot provide a reliable answer because the generated response did not pass citation "
                        "validation after one repair attempt. No unverified draft has been sent."
                    )
                    citation_valid = False
            # Buffer until all citations are checked. The first client-visible token is released after this point.
            visible_first_token_ms = (time.perf_counter() - start) * 1000.0 if answer_text else None

        except DailyBudgetExceeded as exc:
            status = "rate_limited"
            error_kind = "daily_budget"
            answer_text = (
                "I cannot complete this request because the configured daily Groq budget has been exhausted. "
                "I will not substitute an unverified answer. Please retry after the provider's daily reset."
            )
        except ProviderError as exc:
            status = "rate_limited" if exc.status_code == 429 else "error"
            retries += max(0, exc.attempts - 1)
            queue_wait_ms += exc.queue_wait_ms
            retry_wait_ms += exc.retry_wait_ms
            response_headers.update(exc.headers)
            provider_error = str(exc)[:1500]
            lower = str(exc).lower()
            error_kind = "daily_budget" if status == "rate_limited" and ("daily" in lower or "per-day" in lower) else f"provider_{exc.status_code or 'connection'}"
            if status == "rate_limited":
                answer_text = (
                    "I cannot complete this request because the Groq service returned a rate-limit response. "
                    "The request was not answered from unsupported assumptions; please retry after the indicated "
                    "provider reset window."
                )
            else:
                answer_text = "The hosted model request failed before a verified answer was ready. Please retry."
        except (TimeoutError, asyncio.TimeoutError) as exc:
            status = "timeout"
            retries += max(0, getattr(exc, "attempts", 1) - 1)
            queue_wait_ms += getattr(exc, "queue_wait_ms", 0.0)
            retry_wait_ms += getattr(exc, "retry_wait_ms", 0.0)
            error_kind = "deadline_or_model_timeout"
            answer_text = "The request timed out before a verified answer was ready. Please try again."
        except Exception as exc:  # noqa: BLE001 - preserve error data in raw run log
            log.exception("Request %s failed", request_id)
            status = "error"
            error_kind = type(exc).__name__
            answer_text = "An internal error prevented a verified answer from being produced. Please retry."

        total_ms = (time.perf_counter() - start) * 1000.0
        work_ms = max(0.0, total_ms - queue_wait_ms - retry_wait_ms)
        tokens_source = "+".join(sorted(tokens_source_set)) if tokens_source_set else "none"
        cost_usd = (tokens_in * settings.input_price_per_1m + tokens_out * settings.output_price_per_1m) / 1_000_000
        result = {
            "request_id": request_id,
            "question": question,
            "answer": answer_text,
            "status": status,
            "source": "stand-in" if self.llm.backend == "standin" else "real",
            "scenario": scenario or "manual",
            "start_time": start_dt,
            "first_token_ms": round(visible_first_token_ms, 2) if visible_first_token_ms is not None else "",
            "draft_first_token_ms": round(draft_first_token_ms, 2) if draft_first_token_ms is not None else "",
            "total_ms": round(total_ms, 2),
            "retries": retries,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_source": tokens_source,
            "cost_usd": round(cost_usd, 8),
            "client_queue_depth": max(0, int(client_queue_depth)),
            "client_inflight": max(0, int(client_inflight)),
            "retrieval_setting": retrieval_setting,
            "top_score": round(top_score, 4) if top_score is not None else "",
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "paper": c.paper,
                    "citation_name": c.citation_name,
                    "page": c.page,
                    "semantic_score": round(c.semantic_score, 4),
                    "citation": c.citation,
                }
                for c in retrieved
            ],
            "citation_attempts": citation_attempts,
            "retrieval_tool_calls": 1,
            "citation_valid": citation_valid if citation_valid is not None else "",
            "citation_validation_reason": citation_validation_reason,
            "citation_found": citation_found,
            "citation_invalid": citation_invalid,
            "citation_uncited_blocks": citation_uncited_blocks,
            "rewrite_source": rewrite_source if "rewrite_source" in locals() else "not_run",
            "queue_wait_ms": round(queue_wait_ms, 2),
            "provider_queue_ms": round(provider_queue_ms, 2) if provider_queue_ms else "",
            "work_ms": round(work_ms, 2),
            "query_rewrite_ms": round(query_rewrite_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(generation_ms, 2),
            "citation_check_ms": round(citation_check_ms, 2),
            "retry_wait_ms": round(retry_wait_ms, 2),
            "error_kind": error_kind,
            "provider_error": provider_error,
            "rate_limit_limit_requests_rpd": response_headers.get("x-ratelimit-limit-requests", ""),
            "rate_limit_remaining_requests_rpd": response_headers.get("x-ratelimit-remaining-requests", ""),
            "rate_limit_limit_tokens_tpm": response_headers.get("x-ratelimit-limit-tokens", ""),
            "rate_limit_remaining_tokens_tpm": response_headers.get("x-ratelimit-remaining-tokens", ""),
            "retry_after": response_headers.get("retry-after", ""),
            "client_queue_depth": max(0, int(client_queue_depth)),
            "client_inflight": max(0, int(client_inflight)),
            "cost_usd": round(cost_usd, 8),
        }
        await self.recorder.append(
            {
                "request_id": request_id,
                "scenario": scenario or "manual",
                "source": result["source"],
                "start_time": start_dt,
                "first_token_ms": result["first_token_ms"],
                "total_ms": result["total_ms"],
                "status": status,
                "retries": retries,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "retrieval_setting": retrieval_setting,
                "answer_grade": "",
                "queue_wait_ms": result["queue_wait_ms"],
                "provider_queue_ms": result["provider_queue_ms"],
                "work_ms": result["work_ms"],
                "query_rewrite_ms": result["query_rewrite_ms"],
                "retrieval_ms": result["retrieval_ms"],
                "generation_ms": result["generation_ms"],
                "citation_check_ms": result["citation_check_ms"],
                "retry_wait_ms": result["retry_wait_ms"],
                "draft_first_token_ms": result["draft_first_token_ms"],
                "top_score": result["top_score"],
                "citation_attempts": citation_attempts,
                "retrieval_tool_calls": 1,
                "citation_valid": citation_valid if citation_valid is not None else "",
                "citation_validation_reason": citation_validation_reason,
                "citation_invalid": " | ".join(citation_invalid),
                "tokens_source": tokens_source,
                "error_kind": error_kind,
                "provider_error": provider_error,
                "rate_limit_limit_requests_rpd": result["rate_limit_limit_requests_rpd"],
                "rate_limit_remaining_requests_rpd": result["rate_limit_remaining_requests_rpd"],
                "rate_limit_limit_tokens_tpm": result["rate_limit_limit_tokens_tpm"],
                "rate_limit_remaining_tokens_tpm": result["rate_limit_remaining_tokens_tpm"],
                "retry_after": result["retry_after"],
                "client_queue_depth": result["client_queue_depth"],
                "client_inflight": result["client_inflight"],
                "cost_usd": result["cost_usd"],
            }
        )
        return result
