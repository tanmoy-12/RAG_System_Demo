from __future__ import annotations

import argparse
import json

import httpx

from .config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="List model IDs available to this Groq API key.")
    parser.add_argument("--model", default=settings.groq_model, help="Model ID to check")
    args = parser.parse_args()
    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY is empty. Add it to .env first.")
    response = httpx.get(
        f"{settings.groq_base_url}/models",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        timeout=20,
    )
    response.raise_for_status()
    models = response.json().get("data", [])
    ids = sorted(m.get("id", "") for m in models)
    print("Models visible to this API key:")
    for model_id in ids:
        print(" -", model_id)
    print()
    print(json.dumps({"requested_model": args.model, "available_to_key": args.model in ids}, indent=2))
    if args.model not in ids:
        print("Choose a model ID from the list and update GROQ_MODEL in .env.")


if __name__ == "__main__":
    main()
