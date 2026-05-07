#!.venv/bin/python3

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import zvec
from sentence_transformers import SentenceTransformer
from zvec import Collection, CollectionOption, Doc, IVFQueryParam, VectorQuery

from embed.model import build_sentence_transformer
from embed.store import (
    METADATA_FIELD,
    NAME_EMBEDDING_FIELD,
    NAME_FIELD,
    TEXT_EMBEDDING_FIELD,
    TEXT_FIELD,
)


MAX_QUERY_TOP_K = 1024


@dataclass(frozen=True)
class SearchContext:
    collection: Collection
    transformer: SentenceTransformer


def _configure_model_storage(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(models_dir)


def _ensure_zvec_initialized() -> None:
    try:
        zvec.init()
    except RuntimeError as exc:
        if "already" not in str(exc).lower():
            raise


def _collection_path(collection_name: str, zvec_uri: Path) -> Path:
    return zvec_uri / collection_name


def build_context(
    collection_name: str,
    zvec_uri: Path,
    embed_model: str,
    models_dir: Path,
) -> SearchContext:
    _configure_model_storage(models_dir)
    _ensure_zvec_initialized()

    collection_path = _collection_path(collection_name, zvec_uri)
    if not collection_path.exists():
        raise FileNotFoundError(f"Collection not found: {collection_path}")

    collection = zvec.open(
        str(collection_path), option=CollectionOption(read_only=True)
    )
    transformer = build_sentence_transformer(embed_model, models_dir)
    return SearchContext(collection=collection, transformer=transformer)


def _json_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_metadata(value: object) -> object:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _doc_to_result(doc: Doc, score: float | None = None) -> dict:
    result = {
        "id": doc.id,
        "name": doc.field(NAME_FIELD),
        "text": doc.field(TEXT_FIELD),
        "metadata": _parse_metadata(doc.field(METADATA_FIELD)),
    }
    if score is not None:
        result["score"] = score
    elif doc.score is not None:
        result["score"] = float(doc.score)
    return result


def _unique_names(docs: list[Doc], limit: int) -> list[tuple[str, float]]:
    seen: set[str] = set()
    names: list[tuple[str, float]] = []

    for doc in docs:
        name = doc.field(NAME_FIELD)
        if not isinstance(name, str) or not name or name in seen:
            continue
        seen.add(name)
        names.append((name, float(doc.score or 0.0)))
        if len(names) >= limit:
            break

    return names


def _or_equals(field_name: str, values: list[str]) -> str | None:
    clauses = [f"({field_name} = {_json_literal(value)})" for value in values if value]
    if not clauses:
        return None
    return " OR ".join(clauses)


def _run_filter_query(
    collection: Collection,
    filter_value: str | None,
    topk: int,
) -> list[Doc]:
    return collection.query(
        topk=min(max(topk, 1), MAX_QUERY_TOP_K),
        filter=filter_value,
        output_fields=[NAME_FIELD, TEXT_FIELD, METADATA_FIELD],
    )


def _run_all_by_name_filter(
    collection: Collection, filter_value: str | None
) -> list[Doc]:
    return _run_filter_query(collection, filter_value, MAX_QUERY_TOP_K)


def find_by_text_embed(
    ctx: SearchContext,
    query: str,
    top_k: int,
    nprobe: int,
) -> list[dict]:
    query_value = query.strip()
    if not query_value:
        return []

    vector = ctx.transformer.encode([query_value]).tolist()[0]
    docs = ctx.collection.query(
        vectors=VectorQuery(
            field_name=TEXT_EMBEDDING_FIELD,
            vector=vector,
            param=IVFQueryParam(nprobe=nprobe),
        ),
        topk=min(max(top_k, 1), MAX_QUERY_TOP_K),
        output_fields=[NAME_FIELD, TEXT_FIELD, METADATA_FIELD],
    )
    return [_doc_to_result(doc) for doc in docs]


def find_by_text(ctx: SearchContext, query: str, top_k: int) -> list[dict]:
    query_value = query.strip()
    if not query_value:
        return []

    docs = _run_filter_query(
        ctx.collection,
        f"{TEXT_FIELD} LIKE {_json_literal(f'%{query_value}%')}",
        top_k,
    )
    return [_doc_to_result(doc, score=1.0) for doc in docs]


def all_by_name(ctx: SearchContext, query: str) -> list[dict]:
    query_value = query.strip()
    if not query_value:
        return []

    docs = _run_all_by_name_filter(
        ctx.collection,
        f"{NAME_FIELD} LIKE {_json_literal(f'%{query_value}%')}",
    )
    return [_doc_to_result(doc, score=1.0) for doc in docs]


def all_by_name_embed(
    ctx: SearchContext,
    query: str,
    top_k: int,
    nprobe: int,
) -> list[dict]:
    query_value = query.strip()
    if not query_value:
        return []

    vector = ctx.transformer.encode([query_value]).tolist()[0]
    candidate_docs = ctx.collection.query(
        vectors=VectorQuery(
            field_name=NAME_EMBEDDING_FIELD,
            vector=vector,
            param=IVFQueryParam(nprobe=nprobe),
        ),
        topk=min(max(top_k, 1), MAX_QUERY_TOP_K),
        output_fields=[NAME_FIELD],
    )
    candidate_names = _unique_names(candidate_docs, max(top_k, 1))
    if not candidate_names:
        return []

    score_by_name = {name: score for name, score in candidate_names}
    filter_value = _or_equals(NAME_FIELD, [name for name, _ in candidate_names])
    docs = _run_all_by_name_filter(ctx.collection, filter_value)
    return [
        _doc_to_result(doc, score=score_by_name.get(str(doc.field(NAME_FIELD)), 0.0))
        for doc in docs
    ]


def run_search(
    ctx: SearchContext,
    mode: str,
    query: str,
    top_k: int,
    nprobe: int,
) -> list[dict]:
    handlers = {
        "find_by_text_embed": lambda: find_by_text_embed(ctx, query, top_k, nprobe),
        "find_by_text": lambda: find_by_text(ctx, query, top_k),
        "all_by_name": lambda: all_by_name(ctx, query),
        "all_by_name_embed": lambda: all_by_name_embed(ctx, query, top_k, nprobe),
    }

    handler = handlers.get(mode)
    if handler is None:
        raise ValueError(f"Unsupported mode: {mode}")
    return handler()
