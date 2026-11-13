from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import ROOT, settings

ALLOWED = {"correct", "wrong", "refused_correctly", "refused_wrongly"}


def _number(row: dict, key: str) -> float | None:
    try:
        value = row.get(key, "")
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], p: int) -> str:
    return f"{float(np.percentile(values, p)):.1f}" if values else "not measured"


def print_retrieval_summary(rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("retrieval_setting") or "unknown", []).append(row)

    print("\nRetrieval-setting summary for Table 3")
    print("setting | questions | correct+cited % | wrong % | correctly refused % | tokens/answer | first token p50/p95 ms | total p50/p95 ms")
    for setting, items in sorted(grouped.items()):
        graded = [r for r in items if r.get("answer_grade") in ALLOWED]
        total = len(items)
        correct_cited = sum(
            r.get("answer_grade") == "correct" and str(r.get("citation_valid", "")).lower() in {"true", "1"}
            for r in graded
        )
        wrong = sum(r.get("answer_grade") in {"wrong", "refused_wrongly"} for r in graded)
        correctly_refused = sum(r.get("answer_grade") == "refused_correctly" for r in graded)
        token_counts = [
            (a or 0.0) + (b or 0.0)
            for r in items
            for a, b in [(_number(r, "tokens_in"), _number(r, "tokens_out"))]
        ]
        first = [v for r in items if (v := _number(r, "first_token_ms")) is not None]
        elapsed = [v for r in items if (v := _number(r, "total_ms")) is not None]
        pct = lambda value: f"{100.0 * value / max(1, len(graded)):.1f}%"
        mean_tokens = f"{np.mean(token_counts):.1f}" if token_counts else "not measured"
        print(
            f"{setting} | {len({r.get('question_id') for r in items if r.get('question_id')}) or total} "
            f"| {pct(correct_cited)} | {pct(wrong)} | {pct(correctly_refused)} | {mean_tokens} "
            f"| {_percentile(first, 50)}/{_percentile(first, 95)} "
            f"| {_percentile(elapsed, 50)}/{_percentile(elapsed, 95)}"
        )
    print("Percentages use manually graded rows as denominator; ungraded rows are excluded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy manually assigned grades into the required raw results.csv.")
    parser.add_argument("--grading-file", required=True, help="evaluation_*.csv after you fill the grade column")
    parser.add_argument("--results-file", default=str(settings.results_csv))
    args = parser.parse_args()
    grading_path = Path(args.grading_file)
    if not grading_path.is_absolute():
        grading_path = ROOT / grading_path
    results_path = Path(args.results_file)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    with grading_path.open("r", newline="", encoding="utf-8-sig") as f:
        grades = {r["request_id"]: r["grade"].strip() for r in csv.DictReader(f) if r.get("grade", "").strip()}
    invalid = {g for g in grades.values() if g not in ALLOWED}
    if invalid:
        raise SystemExit(f"Invalid grades: {sorted(invalid)}. Allowed: {sorted(ALLOWED)}")
    if not grades:
        raise SystemExit("No grades found. Fill the grade column in the evaluation CSV first.")
    with results_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    updated = 0
    for row in rows:
        if row.get("request_id") in grades:
            row["answer_grade"] = grades[row["request_id"]]
            updated += 1
    temp = results_path.with_suffix(".tmp.csv")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(results_path)
    print(f"Updated answer_grade for {updated} requests in {results_path}")
    print_retrieval_summary([r for r in rows if r.get("request_id") in grades])


if __name__ == "__main__":
    main()
