#!/usr/bin/env .venv/bin/python

from __future__ import annotations

import json
import logging
from pathlib import Path

from embed.cli import (
    DEFAULT_MODELS_DIR,
    normalize_args,
    parse_args,
)
from embed.load import load_documents
from search.search import build_context, run_search


def _get_collection_name(docs_dir: Path) -> str:
    return docs_dir.name or docs_dir.resolve().name or "docs"


def main() -> int:
    args = normalize_args(parse_args())
    docs_dir = Path(args.docs)
    collection_name = _get_collection_name(docs_dir)
    logging.basicConfig(
        level=getattr(logging, args.log),
        format="%(levelname)s: %(message)s",
    )

    if args.query is not None:
        ctx = build_context(
            collection_name=collection_name,
            zvec_uri=Path(args.zvec_uri),
            embed_model=args.embed_model,
            models_dir=DEFAULT_MODELS_DIR,
        )
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
        zvec_uri=Path(args.zvec_uri),
        models_dir=DEFAULT_MODELS_DIR,
        batch_size=args.batch_size,
        chunk_chars=args.chunk_chars,
        max_text_len=args.max_text_len,
        by_source=args.by_source,
        include_ext=args.ext,
    )


if __name__ == "__main__":
    raise SystemExit(main())
