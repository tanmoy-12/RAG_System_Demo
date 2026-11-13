from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    backend: str = os.getenv("LLM_BACKEND", "groq").lower()
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_timeout_seconds: float = _float("OLLAMA_TIMEOUT_SECONDS", 10.0)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    default_retrieval_setting: str = os.getenv("DEFAULT_RETRIEVAL_SETTING", "balanced")
    min_retrieval_score: float = _float("MIN_RETRIEVAL_SCORE", 0.30)
    index_dir: Path = _path("INDEX_DIR", "data/indexes")
    papers_dir: Path = _path("PAPERS_DIR", "data/papers")
    results_csv: Path = _path("RESULTS_CSV", "results/results.csv")
    max_output_tokens: int = _int("MAX_OUTPUT_TOKENS", 700)
    request_deadline_seconds: float = _float("REQUEST_DEADLINE_SECONDS", 50.0)
    model_call_timeout_seconds: float = _float("MODEL_CALL_TIMEOUT_SECONDS", 30.0)
    max_retries: int = _int("MAX_RETRIES", 2)
    retry_max_wait_seconds: float = _float("RETRY_MAX_WAIT_SECONDS", 5.0)
    groq_rpm: int = _int("GROQ_RPM", 0)
    groq_tpm: int = _int("GROQ_TPM", 0)
    groq_rpd: int = _int("GROQ_RPD", 0)
    groq_tpd: int = _int("GROQ_TPD", 0)
    input_price_per_1m: float = _float("GROQ_INPUT_PRICE_PER_1M", 0.0)
    output_price_per_1m: float = _float("GROQ_OUTPUT_PRICE_PER_1M", 0.0)
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = _int("APP_PORT", 8000)


settings = Settings()

RETRIEVAL_SETTINGS = {
    "small": {"chunk_words": 220, "overlap_words": 40, "top_k": 3, "rerank": False},
    "balanced": {"chunk_words": 420, "overlap_words": 80, "top_k": 4, "rerank": True},
    "large": {"chunk_words": 700, "overlap_words": 120, "top_k": 6, "rerank": True},
}
