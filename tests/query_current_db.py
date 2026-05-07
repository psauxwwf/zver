from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embed.cli import (
    DEFAULT_MODELS_DIR,
    MODE_FIND_BY_TEXT_BM25,
    MODE_FIND_BY_TEXT_DENSE,
    MODE_FIND_BY_TEXT_HYBRID,
    MODE_FIND_BY_TEXT_LIKE,
)
from main import _resolve_query_collection_name
from search.search import build_context, run_search
from search.types import search_results_to_jsonable


DOCS_DIR = ROOT / "docs"
ZVEC_DIR = ROOT / "data" / "zvec"
QUERY = "evilginx"
ALL_QUERY_MODES = (
    MODE_FIND_BY_TEXT_DENSE,
    MODE_FIND_BY_TEXT_BM25,
    MODE_FIND_BY_TEXT_HYBRID,
    MODE_FIND_BY_TEXT_LIKE,
    # MODE_ALL_BY_NAME_LIKE,
    # MODE_ALL_BY_NAME_DENSE,
)


def main() -> int:
    failures: list[str] = []

    if not DOCS_DIR.exists():
        print(f"Missing docs directory: {DOCS_DIR}", file=sys.stderr)
        return 1

    if not ZVEC_DIR.exists():
        print(f"Missing zvec directory: {ZVEC_DIR}", file=sys.stderr)
        return 1

    collection_name = _resolve_query_collection_name(DOCS_DIR, ZVEC_DIR, "docs")
    try:
        ctx = build_context(
            collection_name=collection_name,
            zvec_uri=ZVEC_DIR,
            docs_dir=DOCS_DIR,
            embed_model="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            models_dir=DEFAULT_MODELS_DIR,
        )
    except Exception as exc:
        print(f"Failed to build search context: {exc}", file=sys.stderr)
        return 1

    for mode in ALL_QUERY_MODES:
        print(f"\n=== {mode} ===")
        try:
            payload = search_mode(ctx, mode)
        except Exception as exc:
            failures.append(f"{mode}: search failed: {exc}")
            continue

        print(json.dumps(payload, ensure_ascii=False, indent=2))

        if not payload:
            failures.append(f"{mode}: returned no results")
            continue

        if not any(has_query_hit(item) for item in payload):
            failures.append(f"{mode}: no returned chunk mentions {QUERY!r}")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nAll query modes returned matching results.")
    return 0


def search_mode(ctx, mode: str) -> list[dict[str, object]]:
    results = run_search(
        ctx=ctx,
        mode=mode,
        query=QUERY,
        top_k=5,
        nprobe=10,
    )
    payload = search_results_to_jsonable(results)
    if not isinstance(payload, list):
        raise TypeError(f"expected JSON list, got {type(payload).__name__}")
    return payload


def has_query_hit(item: object) -> bool:
    if not isinstance(item, dict):
        return False

    query_lower = QUERY.lower()
    metadata = item.get("metadata")
    metadata_path = ""
    if isinstance(metadata, dict):
        metadata_path = str(metadata.get("path", ""))

    haystacks = (
        str(item.get("name", "")),
        str(item.get("text", "")),
        metadata_path,
    )
    return any(query_lower in value.lower() for value in haystacks)


if __name__ == "__main__":
    raise SystemExit(main())
