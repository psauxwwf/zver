#!.venv/bin/python3

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from docling.document_converter import DocumentConverter
import zvec
from sentence_transformers import SentenceTransformer
from zvec import (
    BM25EmbeddingFunction,
    Collection,
    CollectionOption,
    Doc,
    IVFQueryParam,
    RrfReRanker,
    VectorQuery,
)

from embed.model import build_sentence_transformer
from embed.source import build_hybrid_chunker, get_chunks, get_source_files
from embed.store import (
    METADATA_FIELD,
    NAME_EMBEDDING_FIELD,
    NAME_FIELD,
    TEXT_SPARSE_EMBEDDING_FIELD,
    TEXT_EMBEDDING_FIELD,
    TEXT_FIELD,
    read_embed_config,
)
from .types import SearchResult


MAX_QUERY_TOP_K = 1024


@dataclass
class SearchContext:
    collection: Collection
    transformer: SentenceTransformer
    collection_name: str
    zvec_uri: Path
    docs_dir: Path
    text_bm25_query_encoder: BM25EmbeddingFunction | None = None
    text_bm25_query_encoder_loaded: bool = False


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
) -> BM25EmbeddingFunction | None:
    if not docs_dir.exists() or not docs_dir.is_dir():
        return None

    config = read_embed_config(collection_name, zvec_uri)
    if config is None:
        return None

    chunk_min_chars = int(config.get("chunk_min_chars", 256))
    by_source = bool(config.get("by_source", False))
    include_ext = config.get("include_ext")
    if include_ext is not None and not isinstance(include_ext, list):
        include_ext = None

    converter = DocumentConverter()
    chunker = build_hybrid_chunker(transformer)
    corpus: list[str] = []

    for path in get_source_files(docs_dir, include_ext):
        try:
            corpus.extend(
                get_chunks(
                    path=path,
                    converter=converter,
                    chunker=chunker,
                    by_source=by_source,
                    chunk_min_chars=chunk_min_chars,
                )
            )
        except Exception:
            continue

    if not corpus:
        return None

    return BM25EmbeddingFunction(corpus=corpus, encoding_type="query")


def _ensure_text_bm25_query_encoder(ctx: SearchContext) -> BM25EmbeddingFunction | None:
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
        topk=min(max(topk, 1), MAX_QUERY_TOP_K),
        filter=filter_value,
        output_fields=[NAME_FIELD, TEXT_FIELD, METADATA_FIELD],
    )


def _run_all_by_name_filter(
    collection: Collection, filter_value: str | None
) -> list[Doc]:
    return _run_filter_query(collection, filter_value, MAX_QUERY_TOP_K)


def _run_name_query(collection: Collection) -> list[Doc]:
    return collection.query(
        topk=MAX_QUERY_TOP_K,
        output_fields=[NAME_FIELD],
    )


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
        topk=min(max(top_k, 1), MAX_QUERY_TOP_K),
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

    sparse_vector = encoder.embed(query_value)
    if not sparse_vector:
        return {}
    return sparse_vector


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
        topk=min(max(top_k, 1), MAX_QUERY_TOP_K),
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
    effective_top_k = min(max(top_k, 1), MAX_QUERY_TOP_K)
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


def find_by_text_like(ctx: SearchContext, query: str, top_k: int) -> list[SearchResult]:
    query_value = query.strip()
    if not query_value:
        return []

    docs = _run_filter_query(
        ctx.collection,
        f"{TEXT_FIELD} LIKE {_json_literal(f'%{query_value}%')}",
        top_k,
    )
    return [_doc_to_result(doc, score=1.0) for doc in docs]


def all_by_name_like(ctx: SearchContext, query: str) -> list[SearchResult]:
    query_value = query.strip()
    if not query_value:
        return []

    docs = _run_all_by_name_filter(
        ctx.collection,
        f"{NAME_FIELD} LIKE {_json_literal(f'%{query_value}%')}",
    )
    return [_doc_to_result(doc, score=1.0) for doc in docs]


def all_names(ctx: SearchContext) -> list[str]:
    docs = _run_name_query(ctx.collection)
    names = {
        name
        for doc in docs
        for name in [doc.field(NAME_FIELD)]
        if isinstance(name, str) and name
    }
    return sorted(names)


def all_chunks(ctx: SearchContext) -> list[SearchResult]:
    docs = _run_filter_query(ctx.collection, None, MAX_QUERY_TOP_K)
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
) -> list[SearchResult]:
    handlers = {
        "find_by_text_dense": lambda: find_by_text_dense(ctx, query, top_k, nprobe),
        "find_by_text_bm25": lambda: find_by_text_bm25(ctx, query, top_k),
        "find_by_text_hybrid": lambda: find_by_text_hybrid(ctx, query, top_k, nprobe),
        "find_by_text_like": lambda: find_by_text_like(ctx, query, top_k),
        "all_by_name_like": lambda: all_by_name_like(ctx, query),
        "all_by_name_dense": lambda: all_by_name_dense(ctx, query, top_k, nprobe),
    }

    handler = handlers.get(mode)
    if handler is None:
        raise ValueError(f"Unsupported mode: {mode}")
    return handler()
