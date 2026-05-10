from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embed.cli import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    MODE_ALL_BY_NAME_GREP,
    MODE_FIND_BY_TEXT_GREP,
)
from main import _resolve_query_collection_name  # noqa: E402
from search.search import build_context, run_search  # noqa: E402
from search.types import search_results_to_jsonable  # noqa: E402


# DOCS_DIR = ROOT / "docs"
DOCS_DIR = Path("/") / "home" / "d6f" / "files" / "second-brain.md" / "core"
ZVEC_DIR = ROOT / "data" / "zvec"

TEXT_GREP_CASES = (
    {
        "pattern": r"evilginx",
        "expected_any": ("evilginx",),
    },
    {
        "pattern": r"evilginx2",
        "expected_any": ("evilginx2",),
    },
    {
        "pattern": r"evilginx\|frameless-bitb",
        "expected_any": ("evilginx", "frameless-bitb"),
    },
    {
        "pattern": r"github\.com/.*/evilginx2",
        "expected_any": ("github.com", "evilginx2"),
    },
)

NAME_GREP_CASES = (
    {
        "pattern": r"^Evilginx$",
        "expected_names": ("evilginx",),
    },
    {
        "pattern": r"Acme\|Evilginx",
        "expected_names": ("acme letsencrypt certs", "evilginx"),
    },
    {
        "pattern": r"[Ee][Vv][Ii][Ll][Gg][Ii][Nn][Xx]\|[Ii][Pp][Ss][Ee][Cc]",
        "expected_names": ("evilginx", "ipsec", "l2tp ipsec IKEv2"),
    },
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

    for case in TEXT_GREP_CASES:
        pattern = case["pattern"]
        print(f"\n=== {MODE_FIND_BY_TEXT_GREP}: {pattern!r} ===")
        payload = search_mode(ctx, MODE_FIND_BY_TEXT_GREP, pattern)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not payload:
            failures.append(
                f"{MODE_FIND_BY_TEXT_GREP} {pattern!r}: returned no results"
            )
            continue
        if not any(item_mentions_any(item, case["expected_any"]) for item in payload):
            failures.append(
                f"{MODE_FIND_BY_TEXT_GREP} {pattern!r}: no result mentions any of {case['expected_any']!r}"
            )

    for case in NAME_GREP_CASES:
        pattern = case["pattern"]
        print(f"\n=== {MODE_ALL_BY_NAME_GREP}: {pattern!r} ===")
        payload = search_mode(ctx, MODE_ALL_BY_NAME_GREP, pattern)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not payload:
            failures.append(f"{MODE_ALL_BY_NAME_GREP} {pattern!r}: returned no results")
            continue
        if not any(
            item_has_expected_name(item, case["expected_names"]) for item in payload
        ):
            failures.append(
                f"{MODE_ALL_BY_NAME_GREP} {pattern!r}: no result name matches any of {case['expected_names']!r}"
            )

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("\nAll grep patterns returned matching results.")
    return 0


def search_mode(ctx, mode: str, query: str) -> list[dict[str, object]]:
    results = run_search(
        ctx=ctx,
        mode=mode,
        query=query,
        top_k=5,
        nprobe=10,
    )
    payload = search_results_to_jsonable(results)
    if not isinstance(payload, list):
        raise TypeError(f"expected JSON list, got {type(payload).__name__}")
    return payload


def item_mentions_any(item: object, expected_any: tuple[str, ...]) -> bool:
    if not isinstance(item, dict):
        return False

    metadata = item.get("metadata")
    metadata_path = ""
    if isinstance(metadata, dict):
        metadata_path = str(metadata.get("path", ""))

    haystacks = (
        str(item.get("name", "")).lower(),
        str(item.get("text", "")).lower(),
        metadata_path.lower(),
    )
    expected_lower = tuple(value.lower() for value in expected_any)
    return any(
        expected in haystack for haystack in haystacks for expected in expected_lower
    )


def item_has_expected_name(item: object, expected_names: tuple[str, ...]) -> bool:
    if not isinstance(item, dict):
        return False

    name = str(item.get("name", "")).lower()
    return any(expected.lower() == name for expected in expected_names)


if __name__ == "__main__":
    raise SystemExit(main())
