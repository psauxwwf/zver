#!/usr/bin/env .venv/bin/python

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from embed.cli import (
    DEFAULT_DOCS_DIR,
    DEFAULT_MODELS_DIR,
    normalize_args,
    parse_args,
)
from embed.load import load_documents
from search.search import build_context, run_search


def _get_collection_name(docs_dir: Path) -> str:
    resolved = docs_dir.expanduser().resolve()
    base_name = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in (resolved.name or "docs")
    ).strip("-_.")
    digest = hashlib.sha256(resolved.as_posix().encode()).hexdigest()[:12]
    return f"{(base_name or 'docs')[:48]}-{digest}"


def _list_collection_names(zvec_uri: Path) -> list[str]:
    if not zvec_uri.exists() or not zvec_uri.is_dir():
        return []
    return sorted(path.name for path in zvec_uri.iterdir() if path.is_dir())


def _resolve_query_collection_name(
    docs_dir: Path,
    zvec_uri: Path,
    docs_arg: str,
) -> str:
    requested_collection = _get_collection_name(docs_dir)
    available_collections = _list_collection_names(zvec_uri)

    if docs_arg == str(DEFAULT_DOCS_DIR) and len(available_collections) == 1:
        return available_collections[0]

    return requested_collection


def main() -> int:
    args = normalize_args(parse_args())
    docs_dir = Path(args.docs)
    zvec_uri = Path(args.zvec_uri)
    collection_name = _get_collection_name(docs_dir)
    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(levelname)s: %(message)s",
    )

    if args.query is not None:
        query_collection_name = _resolve_query_collection_name(docs_dir, zvec_uri, args.docs)
        try:
            ctx = build_context(
                collection_name=query_collection_name,
                zvec_uri=zvec_uri,
                embed_model=args.embed_model,
                models_dir=DEFAULT_MODELS_DIR,
            )
        except FileNotFoundError as exc:
            available_collections = _list_collection_names(zvec_uri)
            logging.error("%s", exc)
            logging.error(
                "Query collection is derived from --docs; current --docs=%s maps to collection '%s'",
                docs_dir,
                collection_name,
            )
            if available_collections:
                logging.error("Available collections under %s: %s", args.zvec_uri, ", ".join(available_collections))
            logging.error(
                "Pass the same --docs path you used for ingestion, for example: ./main.py --docs %s --mode=%s --query=%r",
                docs_dir,
                args.mode,
                args.query,
            )
            return 1

        results = run_search(
            ctx=ctx,
            mode=args.mode,
            query=args.query,
            top_k=args.top_k,
            nprobe=args.nprobe,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    return load_documents(
        docs_dir=docs_dir,
        collection_name=collection_name,
        embed_model_name=args.embed_model,
        zvec_uri=zvec_uri,
        models_dir=DEFAULT_MODELS_DIR,
        batch_size=args.batch_size,
        chunk_min_chars=args.chunk_min_chars,
        by_source=args.by_source,
        include_ext=args.ext,
    )


if __name__ == "__main__":
    raise SystemExit(main())
