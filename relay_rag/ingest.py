from __future__ import annotations

import argparse
import json
import logging
import re
import time
import urllib.request
from pathlib import Path

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import RETRIEVAL_SETTINGS, settings
from .papers import PAPERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("relay.ingest")


def download_papers(force: bool = False) -> None:
    settings.papers_dir.mkdir(parents=True, exist_ok=True)
    for paper in PAPERS:
        path = settings.papers_dir / paper["filename"]
        if path.exists() and path.stat().st_size > 1000 and not force:
            log.info("Already present: %s", path.name)
            continue
        request = urllib.request.Request(
            paper["url"], headers={"User-Agent": "TheRelayRAG/0.1 (research assignment)"}
        )
        last_error = None
        for attempt in range(3):
            try:
                log.info("Downloading exact arXiv version %s", paper["arxiv_id"])
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                if not data.startswith(b"%PDF"):
                    raise RuntimeError(f"Expected PDF bytes for {paper['arxiv_id']}; got {data[:20]!r}")
                path.write_bytes(data)
                log.info("Saved %s (%s bytes)", path, len(data))
                break
            except Exception as exc:  # noqa: BLE001 - report download failure with context
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(
                f"Could not download {paper['arxiv_id']}: {last_error}. "
                f"Download {paper['url']} manually and save it as {path.name} in {settings.papers_dir}."
            ) from last_error


def split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    stride = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), stride):
        piece = words[start : start + size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def extract_chunks(chunk_words: int, overlap_words: int) -> list[dict]:
    all_chunks: list[dict] = []
    for paper in PAPERS:
        pdf_path = settings.papers_dir / paper["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing {pdf_path}. Run with --download first.")
        doc = fitz.open(pdf_path)
        for page_index, page in enumerate(doc):
            raw = page.get_text("text") or ""
            clean = re.sub(r"\s+", " ", raw).strip()
            if not clean:
                continue
            page_number = page_index + 1  # PDF page numbering is one-based in citations.
            page_chunks = split_words(clean, chunk_words, overlap_words)
            for local_i, chunk_text in enumerate(page_chunks, start=1):
                chunk_id = f"{paper['arxiv_id']}_p{page_number:04d}_c{local_i:03d}"
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "paper": paper["title"],
                        "citation_name": paper["citation_name"],
                        "arxiv_id": paper["arxiv_id"],
                        "version": paper["arxiv_id"].split("v")[-1],
                        "page": page_number,
                        "source_file": pdf_path.name,
                        "chunk_index_on_page": local_i,
                        "chunk_words": len(chunk_text.split()),
                        "text": chunk_text,
                    }
                )
        doc.close()
        log.info("Extracted %s", paper["arxiv_id"])
    return all_chunks


def build_index(name: str, chunk_words: int, overlap_words: int, model: SentenceTransformer) -> None:
    chunks = extract_chunks(chunk_words, overlap_words)
    if not chunks:
        raise RuntimeError("No text was extracted from the PDFs. Check the files or OCR requirement.")
    log.info("Embedding %d chunks for setting=%s", len(chunks), name)
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    out_dir = settings.index_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    np.save(out_dir / "vectors.npy", vectors)
    (out_dir / "index_meta.json").write_text(
        json.dumps(
            {
                "setting": name,
                "chunk_words": chunk_words,
                "overlap_words": overlap_words,
                "chunk_count": len(chunks),
                "embedding_model": settings.embedding_model,
                "papers": [p["arxiv_id"] for p in PAPERS],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Wrote %d chunks to %s", len(chunks), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fixed arXiv versions and create RAG indexes.")
    parser.add_argument("--download", action="store_true", help="Download exact paper versions from arXiv.")
    parser.add_argument("--force-download", action="store_true", help="Re-download exact versions even if present.")
    parser.add_argument("--settings", nargs="+", choices=list(RETRIEVAL_SETTINGS), default=list(RETRIEVAL_SETTINGS))
    args = parser.parse_args()
    if args.download or args.force_download:
        download_papers(force=args.force_download)
    elif not all((settings.papers_dir / p["filename"]).exists() for p in PAPERS):
        log.info("One or more PDFs missing; downloading the specified versions.")
        download_papers(force=False)
    log.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    for name in args.settings:
        cfg = RETRIEVAL_SETTINGS[name]
        build_index(name, cfg["chunk_words"], cfg["overlap_words"], model)
    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
