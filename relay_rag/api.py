from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import RETRIEVAL_SETTINGS, settings
from .pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("relay.api")
app = FastAPI(title="The Relay RAG API", version="0.1.0")
pipeline: RAGPipeline | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    retrieval_setting: str | None = None
    scenario: str | None = None
    client_queue_depth: int = Field(default=0, ge=0)
    client_inflight: int = Field(default=0, ge=0)


@app.on_event("startup")
async def startup() -> None:
    global pipeline
    pipeline = RAGPipeline()
    log.info("Pipeline ready; backend=%s", settings.backend)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "backend": settings.backend,
        "model": settings.groq_model if settings.backend == "groq" else "standin-chat-model",
        "retrieval_settings": list(RETRIEVAL_SETTINGS),
    }


@app.post("/ask")
async def ask(request: AskRequest) -> dict:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")
    try:
        return await pipeline.answer(
            request.question,
            retrieval_setting=request.retrieval_setting,
            scenario=request.scenario,
            client_queue_depth=request.client_queue_depth,
            client_inflight=request.client_inflight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Validate the complete draft first, then stream only the accepted answer to the caller."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")
    try:
        result = await pipeline.answer(
            request.question,
            retrieval_setting=request.retrieval_setting,
            scenario=request.scenario,
            client_queue_depth=request.client_queue_depth,
            client_inflight=request.client_inflight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def events():
        meta = {k: result.get(k) for k in ("request_id", "status", "first_token_ms", "total_ms", "citation_valid")}
        yield f"event: metadata\ndata: {json.dumps(meta)}\n\n"
        words = result["answer"].split()
        for i in range(0, len(words), 5):
            piece = " ".join(words[i : i + 5]) + (" " if i + 5 < len(words) else "")
            yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"
            await asyncio.sleep(0.015)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def main() -> None:
    import uvicorn
    uvicorn.run("relay_rag.api:app", host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    main()
