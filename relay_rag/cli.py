from __future__ import annotations

import argparse
import asyncio
import json

from .pipeline import RAGPipeline


async def run(question: str, setting: str, scenario: str) -> None:
    pipeline = RAGPipeline()
    result = await pipeline.answer(question, retrieval_setting=setting, scenario=scenario)
    print("\n" + result["answer"] + "\n")
    print(json.dumps({k: result.get(k) for k in (
        "status", "source", "retrieval_setting", "top_score", "first_token_ms", "total_ms",
        "retries", "tokens_in", "tokens_out", "citation_valid",
        "citation_validation_reason", "citation_found", "citation_invalid",
        "citation_uncited_blocks", "error_kind", "request_id"
    )}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the three indexed research papers.")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--setting", choices=["small", "balanced", "large"], default=None)
    parser.add_argument("--scenario", default="manual")
    args = parser.parse_args()
    asyncio.run(run(args.question, args.setting, args.scenario))


if __name__ == "__main__":
    main()
