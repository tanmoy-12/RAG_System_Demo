from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from .config import RETRIEVAL_SETTINGS, settings

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "what", "why", "how", "when", "where",
    "which", "who", "does", "do", "did", "can", "could", "would", "should", "this", "that",
    "these", "those", "it", "its", "as", "at", "about", "into", "than", "then", "their", "they",
}


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper: str
    citation_name: str
    arxiv_id: str
    page: int
    text: str
    semantic_score: float
    final_score: float

    @property
    def citation(self) -> str:
        return f"[{self.citation_name}, p. {self.page}]"


class RetrieverRegistry:
    """Lazily load each retrieval setting while sharing one embedding model."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.indices: dict[str, tuple[list[dict], np.ndarray]] = {}

    def _embedding_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.embedding_model, device="cpu")
        return self.model

    def _load_index(self, setting_name: str) -> tuple[list[dict], np.ndarray]:
        if setting_name not in RETRIEVAL_SETTINGS:
            raise ValueError(f"Unknown retrieval setting: {setting_name}")
        if setting_name not in self.indices:
            index_dir = settings.index_dir / setting_name
            chunks_path = index_dir / "chunks.jsonl"
            vectors_path = index_dir / "vectors.npy"
            if not chunks_path.exists() or not vectors_path.exists():
                raise FileNotFoundError(
                    f"Index '{setting_name}' is missing. Run: python -m relay_rag.ingest --download"
                )
            with chunks_path.open("r", encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f if line.strip()]
            vectors = np.load(vectors_path).astype("float32")
            if len(chunks) != vectors.shape[0]:
                raise RuntimeError(f"Index corruption: {len(chunks)} chunks but {vectors.shape[0]} vectors.")
            self.indices[setting_name] = (chunks, vectors)
        return self.indices[setting_name]

    def search(
        self,
        query_variants: list[str],
        setting_name: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> tuple[list[RetrievedChunk], float]:
        setting_name = setting_name or settings.default_retrieval_setting
        cfg = RETRIEVAL_SETTINGS[setting_name]
        top_k = top_k or cfg["top_k"]
        rerank = cfg["rerank"] if rerank is None else rerank
        chunks, vectors = self._load_index(setting_name)
        model = self._embedding_model()
        variants = [q.strip() for q in query_variants if q and q.strip()]
        if not variants:
            raise ValueError("At least one non-empty query is required.")
        q_vectors = model.encode(
            variants, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        sims = q_vectors @ vectors.T
        semantic = np.max(sims, axis=0)
        # Deterministic, transparent lexical reranking; gating still uses semantic similarity only.
        query_terms = {
            t for variant in variants for t in re.findall(r"[a-z0-9]+", variant.lower()) if t not in STOPWORDS
        }
        lexical = np.zeros(len(chunks), dtype="float32")
        if query_terms:
            for i, chunk in enumerate(chunks):
                chunk_terms = set(re.findall(r"[a-z0-9]+", chunk["text"].lower()))
                lexical[i] = len(query_terms & chunk_terms) / max(1, len(query_terms))
        if rerank:
            final = (0.85 * semantic) + (0.15 * lexical)
        else:
            final = semantic.copy()
        order = np.argsort(final)[::-1][: min(top_k, len(chunks))]
        found = [
            RetrievedChunk(
                chunk_id=chunks[int(i)]["chunk_id"],
                paper=chunks[int(i)]["paper"],
                citation_name=chunks[int(i)]["citation_name"],
                arxiv_id=chunks[int(i)]["arxiv_id"],
                page=int(chunks[int(i)]["page"]),
                text=chunks[int(i)]["text"],
                semantic_score=float(semantic[int(i)]),
                final_score=float(final[int(i)]),
            )
            for i in order
        ]
        best_semantic = float(np.max(semantic)) if semantic.size else -1.0
        return found, best_semantic
