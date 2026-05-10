#!.venv/bin/python3

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import dashtext
import pygnuregex
import zvec
from sentence_transformers import SentenceTransformer
from zvec import (
    Collection,
    CollectionOption,
    Doc,
    IVFQueryParam,
    RrfReRanker,
    VectorQuery,
)

from embed.model import build_sentence_transformer
from embed.store import (
    BM25_ENCODER_FILENAME,
    METADATA_FIELD,
    NAME_EMBEDDING_FIELD,
    NAME_FIELD,
    read_doc_manifest,
    TEXT_SPARSE_EMBEDDING_FIELD,
    TEXT_EMBEDDING_FIELD,
    TEXT_FIELD,
)
from .types import SearchResult


@dataclass
class SearchContext:
    collection: Collection
    transformer: SentenceTransformer
    collection_name: str
    zvec_uri: Path
    docs_dir: Path
    text_bm25_query_encoder: dashtext.SparseVectorEncoder | None = None
    text_bm25_query_encoder_loaded: bool = False
    doc_manifest_ids: list[str] | None = None
    doc_manifest_ids_loaded: bool = False


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
    docs_dir: Path,
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
    return SearchContext(
        collection=collection,
        transformer=transformer,
        collection_name=collection_name,
        zvec_uri=zvec_uri,
        docs_dir=docs_dir,
    )


def _build_text_bm25_query_encoder(
    collection_name: str,
    zvec_uri: Path,
    docs_dir: Path,
    transformer: SentenceTransformer,
) -> dashtext.SparseVectorEncoder | None:
    encoder_path = zvec_uri / collection_name / BM25_ENCODER_FILENAME
    if not encoder_path.exists():
        return None

    encoder = dashtext.SparseVectorEncoder(b=0.75, k1=1.2)
    encoder.load(str(encoder_path))
    return encoder


def _ensure_text_bm25_query_encoder(
    ctx: SearchContext,
) -> dashtext.SparseVectorEncoder | None:
    if ctx.text_bm25_query_encoder_loaded:
        return ctx.text_bm25_query_encoder

    ctx.text_bm25_query_encoder = _build_text_bm25_query_encoder(
        collection_name=ctx.collection_name,
        zvec_uri=ctx.zvec_uri,
        docs_dir=ctx.docs_dir,
        transformer=ctx.transformer,
    )
    ctx.text_bm25_query_encoder_loaded = True
    return ctx.text_bm25_query_encoder


def _json_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_metadata(value: object) -> object:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _doc_to_result(doc: Doc, score: float | None = None) -> SearchResult:
    result_score = score if score is not None else float(doc.score or 0.0)
    return SearchResult.model_validate(
        {
            "id": doc.id,
            "name": doc.field(NAME_FIELD),
            "text": doc.field(TEXT_FIELD),
            "metadata": _parse_metadata(doc.field(METADATA_FIELD)),
            "score": result_score,
        }
    )


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
        topk=max(topk, 1),
        filter=filter_value,
        output_fields=[NAME_FIELD, TEXT_FIELD, METADATA_FIELD],
    )


def _collection_doc_count(collection: Collection) -> int:
    return max(int(collection.stats.doc_count), 1)


def _compile_grep_pattern(query_value: str) -> pygnuregex.Pattern:
    try:
        return pygnuregex.compile(
            query_value.encode("utf-8"),
            int(pygnuregex.SyntaxFlag.RE_SYNTAX_GREP),
        )
    except RuntimeError as exc:
        raise ValueError(f"Invalid grep pattern: {exc}") from exc


def _grep_matches(pattern: pygnuregex.Pattern, value: str) -> bool:
    return pattern.search(value.encode("utf-8")) >= 0


def _build_doc_manifest_ids(
    collection_name: str,
    zvec_uri: Path,
) -> list[str] | None:
    manifest = read_doc_manifest(collection_name=collection_name, zvec_uri=zvec_uri)
    if manifest is None:
        return None
    return [doc_id for doc_ids in manifest.values() for doc_id in doc_ids]


def _ensure_doc_manifest_ids(ctx: SearchContext) -> list[str] | None:
    if ctx.doc_manifest_ids_loaded:
        return ctx.doc_manifest_ids

    ctx.doc_manifest_ids = _build_doc_manifest_ids(
        collection_name=ctx.collection_name,
        zvec_uri=ctx.zvec_uri,
    )
    ctx.doc_manifest_ids_loaded = True
    return ctx.doc_manifest_ids


def _iter_all_docs(ctx: SearchContext, batch_size: int = 1000) -> Iterator[Doc]:
    doc_ids = _ensure_doc_manifest_ids(ctx)
    if doc_ids is None:
        logging.warning(
            "Doc manifest is missing for collection '%s'; falling back to a full filter query based on doc_count. Re-run embedding to enable fetch-based full scans.",
            ctx.collection_name,
        )
        yield from _run_filter_query(
            ctx.collection,
            None,
            _collection_doc_count(ctx.collection),
        )
        return

    effective_batch_size = max(batch_size, 1)
    for start in range(0, len(doc_ids), effective_batch_size):
        batch_ids = doc_ids[start : start + effective_batch_size]
        docs_by_id = ctx.collection.fetch(batch_ids)
        for doc_id in batch_ids:
            doc = docs_by_id.get(doc_id)
            if doc is not None:
                yield doc


def find_by_text_dense(
    ctx: SearchContext,
    query: str,
    top_k: int,
    nprobe: int,
) -> list[SearchResult]:
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
        topk=max(top_k, 1),
        output_fields=[NAME_FIELD, TEXT_FIELD, METADATA_FIELD],
    )
    return [_doc_to_result(doc) for doc in docs]


def _build_text_bm25_query_vector(
    ctx: SearchContext,
    query_value: str,
) -> dict[int, float]:
    if ctx.collection.schema.vector(TEXT_SPARSE_EMBEDDING_FIELD) is None:
        raise RuntimeError(
            "BM25 search is unavailable for this collection. Re-run embedding with the current version to build sparse vectors."
        )

    encoder = _ensure_text_bm25_query_encoder(ctx)
    if encoder is None:
        raise RuntimeError(
            "BM25 search is unavailable for this collection. Re-run embedding with the current version to build sparse vectors."
        )

    sparse_vector = encoder.encode_queries(query_value)
    if isinstance(sparse_vector, list):
        raise TypeError("Expected a single sparse vector for one query string")
    if not sparse_vector:
        return {}
    return {
        int(key): float(value)
        for key, value in sparse_vector.items()
        if float(value) > 0
    }


def find_by_text_bm25(ctx: SearchContext, query: str, top_k: int) -> list[SearchResult]:
    query_value = query.strip()
    if not query_value:
        return []

    sparse_vector = _build_text_bm25_query_vector(ctx, query_value)
    if not sparse_vector:
        return []

    docs = ctx.collection.query(
        vectors=VectorQuery(
            field_name=TEXT_SPARSE_EMBEDDING_FIELD,
            vector=sparse_vector,
        ),
        topk=max(top_k, 1),
        output_fields=[NAME_FIELD, TEXT_FIELD, METADATA_FIELD],
    )
    return [_doc_to_result(doc) for doc in docs]


def find_by_text_hybrid(
    ctx: SearchContext,
    query: str,
    top_k: int,
    nprobe: int,
) -> list[SearchResult]:
    query_value = query.strip()
    if not query_value:
        return []

    sparse_vector = _build_text_bm25_query_vector(ctx, query_value)
    if not sparse_vector:
        return find_by_text_dense(ctx, query_value, top_k, nprobe)

    dense_vector = ctx.transformer.encode([query_value]).tolist()[0]
    effective_top_k = max(top_k, 1)
    docs = ctx.collection.query(
        vectors=[
            VectorQuery(
                field_name=TEXT_EMBEDDING_FIELD,
                vector=dense_vector,
                param=IVFQueryParam(nprobe=nprobe),
            ),
            VectorQuery(
                field_name=TEXT_SPARSE_EMBEDDING_FIELD,
                vector=sparse_vector,
            ),
        ],
        topk=effective_top_k,
        reranker=RrfReRanker(topn=effective_top_k),
        output_fields=[NAME_FIELD, TEXT_FIELD, METADATA_FIELD],
    )
    return [_doc_to_result(doc) for doc in docs]


def find_by_text_grep(ctx: SearchContext, query: str, top_k: int) -> list[SearchResult]:
    query_value = query.strip()
    if not query_value:
        return []

    pattern = _compile_grep_pattern(query_value)
    docs: list[Doc] = []
    effective_top_k = max(top_k, 1)

    for doc in _iter_all_docs(ctx):
        if not isinstance(doc.field(TEXT_FIELD), str):
            continue
        if not _grep_matches(pattern, str(doc.field(TEXT_FIELD))):
            continue
        docs.append(doc)
        if len(docs) >= effective_top_k:
            break

    return [_doc_to_result(doc, score=1.0) for doc in docs]


def all_by_name_grep(ctx: SearchContext, query: str) -> list[SearchResult]:
    query_value = query.strip()
    if not query_value:
        return []

    pattern = _compile_grep_pattern(query_value)
    docs = [
        doc
        for doc in _iter_all_docs(ctx)
        if isinstance(doc.field(NAME_FIELD), str)
        and _grep_matches(pattern, str(doc.field(NAME_FIELD)))
    ]
    return [_doc_to_result(doc, score=1.0) for doc in docs]


def all_names(ctx: SearchContext) -> list[str]:
    names = {
        name
        for doc in _iter_all_docs(ctx)
        for name in [doc.field(NAME_FIELD)]
        if isinstance(name, str) and name
    }
    return sorted(names)


def all_chunks(ctx: SearchContext) -> list[SearchResult]:
    docs = list(_iter_all_docs(ctx))
    return [_doc_to_result(doc, score=1.0) for doc in docs]


def all_by_name_dense(
    ctx: SearchContext,
    query: str,
    top_k: int,
    nprobe: int,
) -> list[SearchResult]:
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
        topk=max(top_k, 1),
        output_fields=[NAME_FIELD],
    )
    candidate_names = _unique_names(candidate_docs, max(top_k, 1))
    if not candidate_names:
        return []

    score_by_name = {name: score for name, score in candidate_names}
    candidate_name_set = set(score_by_name)
    docs = [
        doc
        for doc in _iter_all_docs(ctx)
        if str(doc.field(NAME_FIELD)) in candidate_name_set
    ]
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
) -> list[SearchResult]:
    handlers = {
        "find_by_text_dense": lambda: find_by_text_dense(ctx, query, top_k, nprobe),
        "find_by_text_bm25": lambda: find_by_text_bm25(ctx, query, top_k),
        "find_by_text_hybrid": lambda: find_by_text_hybrid(ctx, query, top_k, nprobe),
        "find_by_text_grep": lambda: find_by_text_grep(ctx, query, top_k),
        "all_by_name_grep": lambda: all_by_name_grep(ctx, query),
        "all_by_name_dense": lambda: all_by_name_dense(ctx, query, top_k, nprobe),
    }

    handler = handlers.get(mode)
    if handler is None:
        raise ValueError(f"Unsupported mode: {mode}")
    return handler()
