from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from .config import ROOT, settings

SCENARIOS = {
    "steady": {"requests": 20, "concurrency": 2, "interval_seconds": 0.45},
    "burst": {"requests": 30, "concurrency": 30, "interval_seconds": 0.0},
    "sustained_overload": {"requests": 60, "concurrency": 12, "interval_seconds": 0.08},
    "rpm_limit_halved": {"requests": 25, "concurrency": 8, "interval_seconds": 0.0},
    "daily_budget_exhausted": {"requests": 12, "concurrency": 6, "interval_seconds": 0.0},
}


def percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


async def configure_standin(client: httpx.AsyncClient, scenario: str) -> None:
    state_resp = await client.get("http://127.0.0.1:8001/admin/state")
    state_resp.raise_for_status()
    config = state_resp.json()["config"]
    # Preserve calibrated latency/jitter/failure settings from the stand-in's current config.
    config["rpm_limit"] = 0
    config["tpm_limit"] = 0
    config["daily_limit"] = 0
    config["daily_token_limit"] = 0
    config["max_inflight"] = 0
    if scenario == "rpm_limit_halved":
        # Synthetic baseline is 12 RPM; this scenario deliberately halves it to 6 RPM.
        config["rpm_limit"] = 6
        config["failure_rate"] = 0.0
    elif scenario == "daily_budget_exhausted":
        config["daily_limit"] = 5
        config["failure_rate"] = 0.0
    elif scenario == "sustained_overload":
        config["per_extra_inflight_ms"] = 300
    elif scenario == "burst":
        config["per_extra_inflight_ms"] = 260
    response = await client.post("http://127.0.0.1:8001/admin/config", json=config)
    response.raise_for_status()
    reset = await client.post("http://127.0.0.1:8001/admin/reset")
    reset.raise_for_status()


async def run_scenario(
    scenario: str,
    source: str,
    api_url: str,
    n: int,
    concurrency: int,
    interval: float,
    retrieval_setting: str,
) -> dict:
    run_id = str(uuid.uuid4())[:8]
    client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))
    health = await client.get(f"{api_url}/health")
    health.raise_for_status()
    actual_backend = health.json().get("backend", "unknown")
    expected_backend = "standin" if source == "stand-in" else "groq"
    if actual_backend != expected_backend:
        await client.aclose()
        raise RuntimeError(
            f"API backend is '{actual_backend}', but --source={source}. Restart API with matching LLM_BACKEND."
        )
    if source == "stand-in":
        await configure_standin(client, scenario)

    semaphore = asyncio.Semaphore(max(1, concurrency))
    waiting = 0
    active = 0
    peak_queue = 0
    queue_lock = asyncio.Lock()
    results: list[dict] = []
    started = time.perf_counter()

    async def one(index: int) -> None:
        nonlocal waiting, active, peak_queue
        if interval > 0:
            await asyncio.sleep(index * interval)
        async with queue_lock:
            waiting += 1
            peak_queue = max(peak_queue, max(0, waiting - (concurrency - active)))
        async with semaphore:
            async with queue_lock:
                waiting -= 1
                active += 1
                request_queue_depth = waiting
                request_inflight = active
            try:
                payload = {
                    "question": (
                        "Explain the main contribution of the paper and identify the evidence in the retrieved passages."
                    ),
                    "retrieval_setting": retrieval_setting,
                    "scenario": scenario,
                    "client_queue_depth": request_queue_depth,
                    "client_inflight": request_inflight,
                }
                t0 = time.perf_counter()
                try:
                    response = await client.post(f"{api_url}/ask", json=payload)
                    elapsed = (time.perf_counter() - t0) * 1000
                    response.raise_for_status()
                    item = response.json()
                    item["load_http_elapsed_ms"] = round(elapsed, 2)
                except httpx.HTTPStatusError as exc:
                    elapsed = (time.perf_counter() - t0) * 1000
                    item = {
                        "status": "error",
                        "error": str(exc),
                        "load_http_elapsed_ms": round(elapsed, 2),
                    }
                except Exception as exc:  # noqa: BLE001
                    elapsed = (time.perf_counter() - t0) * 1000
                    item = {"status": "error", "error": str(exc), "load_http_elapsed_ms": round(elapsed, 2)}
                item["load_index"] = index
                results.append(item)
            finally:
                async with queue_lock:
                    active -= 1

    await asyncio.gather(*(one(i) for i in range(n)))
    duration = time.perf_counter() - started
    await client.aclose()

    ok = [r for r in results if r.get("status") == "ok"]
    limited = [r for r in results if r.get("status") == "rate_limited"]
    timeouts = [r for r in results if r.get("status") == "timeout"]
    errors = [r for r in results if r.get("status") == "error"]
    first = [float(r["first_token_ms"]) for r in ok if r.get("first_token_ms") not in (None, "")]
    total = [
        float(r.get("total_ms") or r.get("load_http_elapsed_ms"))
        for r in results
        if r.get("total_ms") not in (None, "") or r.get("load_http_elapsed_ms") not in (None, "")
    ]
    token_counts = [int(r.get("tokens_in", 0)) + int(r.get("tokens_out", 0)) for r in ok]
    total_in = sum(int(r.get("tokens_in", 0)) for r in ok)
    total_out = sum(int(r.get("tokens_out", 0)) for r in ok)
    total_cost_usd = (total_in * settings.input_price_per_1m + total_out * settings.output_price_per_1m) / 1_000_000
    cost_per_1000 = (total_cost_usd / len(ok) * 1000) if ok else None
    summary = {
        "run_id": run_id,
        "scenario": scenario,
        "source": source,
        "requests_sent": len(results),
        "duration_seconds": round(duration, 3),
        "offered_rate_per_minute": round(len(results) / max(duration, 0.001) * 60, 2),
        "succeeded_percent": round(len(ok) / max(1, len(results)) * 100, 1),
        "rate_limited_percent": round(len(limited) / max(1, len(results)) * 100, 1),
        "timed_out_percent": round(len(timeouts) / max(1, len(results)) * 100, 1),
        "errored_percent": round(len(errors) / max(1, len(results)) * 100, 1),
        "timed_out_or_given_up_percent": round((len(timeouts) + len(errors)) / max(1, len(results)) * 100, 1),
        "first_token_p50_p95_p99_ms": [percentile(first, p) for p in (50, 95, 99)],
        "total_p50_p95_p99_ms": [percentile(total, p) for p in (50, 95, 99)],
        "retries_per_request": round(sum(int(r.get("retries", 0)) for r in results) / max(1, len(results)), 2),
        "peak_client_queue_length": max(
            [int(r.get("client_queue_depth", 0)) for r in results] + [peak_queue]
        ),
        "peak_client_inflight": max([int(r.get("client_inflight", 0)) for r in results] or [0]),
        "tokens_per_successful_answer": round(statistics.mean(token_counts), 1) if token_counts else None,
        "cost_per_1000_answers_usd": round(cost_per_1000, 6) if cost_per_1000 is not None else None,
        "results": results,
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"loadtest_{scenario}_{run_id}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Raw scenario detail: {out}")
    return summary


async def main_async(args) -> None:
    api_url = args.api_url.rstrip("/")
    if args.scenario == "all":
        scenarios = list(SCENARIOS)
    else:
        scenarios = [args.scenario]
    all_summaries = []
    for scenario in scenarios:
        defaults = SCENARIOS[scenario]
        n = args.requests or defaults["requests"]
        concurrency = args.concurrency or defaults["concurrency"]
        interval = args.interval if args.interval is not None else defaults["interval_seconds"]
        all_summaries.append(
            await run_scenario(
                scenario, args.source, api_url, n, concurrency, interval, args.retrieval_setting
            )
        )
    table_rows = []
    for summary in all_summaries:
        table_rows.append(
            {
                "scenario": summary["scenario"],
                "source": summary["source"],
                "requests_sent": summary["requests_sent"],
                "duration_seconds": summary["duration_seconds"],
                "offered_rate_per_minute": summary["offered_rate_per_minute"],
                "first_token_p50_p95_p99_ms": summary["first_token_p50_p95_p99_ms"],
                "total_p50_p95_p99_ms": summary["total_p50_p95_p99_ms"],
                "succeeded_percent": summary["succeeded_percent"],
                "rate_limited_percent": summary["rate_limited_percent"],
                "timed_out_or_given_up_percent": summary["timed_out_or_given_up_percent"],
                "retries_per_request": summary["retries_per_request"],
                "peak_client_queue_length": summary["peak_client_queue_length"],
                "peak_client_inflight": summary["peak_client_inflight"],
                "tokens_per_successful_answer": summary["tokens_per_successful_answer"],
                "cost_per_1000_answers_usd": summary["cost_per_1000_answers_usd"],
            }
        )
    out_path = ROOT / "results" / f"loadtest_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Summary table: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test the Relay API against real Groq or local stand-in.")
    parser.add_argument("--source", choices=["real", "stand-in"], required=True)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="steady")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=0, help="Override scenario request count")
    parser.add_argument("--concurrency", type=int, default=0, help="Override scenario concurrency")
    parser.add_argument("--interval", type=float, default=None, help="Seconds between scheduled arrivals")
    parser.add_argument("--retrieval-setting", choices=["small", "balanced", "large"], default="balanced")
    args = parser.parse_args()
    if args.source == "real":
        if args.scenario != "steady" or args.requests > 10 or args.concurrency > 2:
            parser.error("Real Groq probes are deliberately capped: use --scenario steady, <=10 requests, concurrency <=2. Use stand-in for heavy loads.")
        if args.requests == 0:
            args.requests = 8
        if args.concurrency == 0:
            args.concurrency = 1
        if args.interval is None:
            args.interval = 1.0
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
