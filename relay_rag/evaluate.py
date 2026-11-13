from __future__ import annotations

import argparse
import asyncio
import csv
import json
import uuid
from pathlib import Path

import httpx

from .config import ROOT


async def run_evaluation(args) -> None:
    bank_path = Path(args.questions).resolve() if args.questions else Path(__file__).with_name("questions.json")
    questions = json.loads(bank_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    if len(questions) < 15 and args.mode == "quality":
        raise SystemExit("Quality mode requires at least 15 questions. Use the 18-question starter bank or add your own.")
    settings_list = args.settings
    if args.mode == "quality" and len(settings_list) != 1:
        raise SystemExit("Quality mode uses one selected setting; use retrieval mode for three-setting comparison.")
    if args.mode == "retrieval" and len(settings_list) < 3:
        raise SystemExit("Retrieval mode must compare at least three settings: small, balanced, large.")

    run_id = str(uuid.uuid4())[:8]
    base_url = args.api_url.rstrip("/")
    out_rows = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        health = await client.get(f"{base_url}/health")
        health.raise_for_status()
        for setting in settings_list:
            for question in questions:
                response = await client.post(
                    f"{base_url}/ask",
                    json={
                        "question": question["question"],
                        "retrieval_setting": setting,
                        "scenario": f"{args.mode}_{setting}",
                    },
                )
                response.raise_for_status()
                result = response.json()
                out_rows.append(
                    {
                        "run_id": run_id,
                        "request_id": result.get("request_id", ""),
                        "question_id": question["id"],
                        "target_paper": question["paper"],
                        "question": question["question"],
                        "expected_evidence_notes": question["expected_evidence"],
                        "retrieval_setting": setting,
                        "status": result.get("status", ""),
                        "answer": result.get("answer", ""),
                        "citations": json.dumps(result.get("retrieved_chunks", []), ensure_ascii=False),
                        "top_score": result.get("top_score", ""),
                        "tokens_in": result.get("tokens_in", 0),
                        "tokens_out": result.get("tokens_out", 0),
                        "first_token_ms": result.get("first_token_ms", ""),
                        "total_ms": result.get("total_ms", ""),
                        "citation_valid": result.get("citation_valid", ""),
                        "grade": "",
                        "grader_notes": "",
                    }
                )
                print(f"[{len(out_rows)}/{len(questions) * len(settings_list)}] {question['id']} / {setting} -> {result.get('status')}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"evaluation_{args.mode}_{run_id}.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nEvaluation and manual grading sheet: {out}")
    print("Grade each row as: correct, wrong, refused_correctly, or refused_wrongly.")
    print("Then run: python -m relay_rag.grade_results --grading-file results\\evaluation_....csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer-quality or retrieval-setting evaluation against /ask.")
    parser.add_argument("--mode", choices=["quality", "retrieval"], default="quality")
    parser.add_argument("--settings", nargs="+", choices=["small", "balanced", "large"], default=None)
    parser.add_argument("--limit", type=int, default=0, help="Question count; quality mode needs >=15")
    parser.add_argument("--questions", default=None, help="Optional custom JSON question bank")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.settings is None:
        args.settings = ["balanced"] if args.mode == "quality" else ["small", "balanced", "large"]
    if args.mode == "retrieval" and args.limit == 0:
        args.limit = 5  # paired 5-question pilot across all three retrieval configurations
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
