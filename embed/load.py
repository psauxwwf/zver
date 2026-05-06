from __future__ import annotations

import logging
import os
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sentence_transformers import SentenceTransformer
from zvec import Collection, Doc

from .document import (
    build_docs,
    normalize_source_name,
)
from .store import (
    get_or_create_collection,
)
from .source import (
    _split_with_max,
    convert_file,
    get_source_files,
    merge_chunks,
)


def _build_chunker(transformer: SentenceTransformer) -> HybridChunker:
    tokenizer = HuggingFaceTokenizer(tokenizer=transformer.tokenizer)
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def _flush_batch(collection: Collection, docs_batch: list[Doc]) -> int:
    if not docs_batch:
        return 0

    collection.insert(docs_batch)
    inserted = len(docs_batch)
    docs_batch.clear()
    return inserted


def _configure_model_storage(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(models_dir)


def _build_transformer(
    embed_model_name: str,
    models_dir: Path,
) -> tuple[SentenceTransformer, int]:
    transformer = SentenceTransformer(
        embed_model_name,
        cache_folder=str(models_dir),
    )
    dim = transformer.get_embedding_dimension()
    if dim is None:
        raise RuntimeError(
            f"Embedding dimension is undefined for model {embed_model_name}"
        )
    return transformer, dim


def _get_chunks(
    path: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
    by_source: bool,
    chunk_chars: int,
    max_text_len: int,
) -> list[str]:
    chunks = convert_file(path, converter, chunker)

    if by_source:
        full_text = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
        if not full_text:
            return []
        return _split_with_max(full_text, max_text_len)

    return merge_chunks(chunks, chunk_chars, max_text_len)


def _extend_docs_batch(
    docs_batch: list[Doc],
    path: Path,
    chunks: list[str],
    transformer: SentenceTransformer,
) -> None:
    text_embeddings = transformer.encode(chunks).tolist()
    name = normalize_source_name(path)
    name_embedding = transformer.encode([name]).tolist()[0]
    docs_batch.extend(build_docs(path, chunks, text_embeddings, name_embedding))


def load_documents(
    docs_dir: Path,
    collection_name: str,
    embed_model_name: str,
    zvec_uri: Path,
    models_dir: Path,
    batch_size: int,
    chunk_chars: int,
    max_text_len: int,
    by_source: bool,
    include_ext: list[str] | None,
) -> int:
    _configure_model_storage(models_dir)
    converter = DocumentConverter()
    transformer, dim = _build_transformer(embed_model_name, models_dir)
    chunker = _build_chunker(transformer)
    collection = get_or_create_collection(
        collection_name=collection_name,
        zvec_uri=str(zvec_uri),
        dim=dim,
    )

    files = list(get_source_files(docs_dir, include_ext))
    if not files:
        logging.warning("No files found under %s", docs_dir)
        return 1

    docs_batch: list[Doc] = []
    ingested = 0
    skipped = 0

    for path in files:
        try:
            chunks = _get_chunks(
                path=path,
                converter=converter,
                chunker=chunker,
                by_source=by_source,
                chunk_chars=chunk_chars,
                max_text_len=max_text_len,
            )
        except Exception as exc:  # pragma: no cover - logging only
            skipped += 1
            logging.warning("Skipping %s: %s", path, exc)
            continue

        if not chunks:
            logging.debug("No chunks produced for %s", path)
            continue

        _extend_docs_batch(docs_batch, path, chunks, transformer)

        if batch_size > 0 and len(docs_batch) >= batch_size:
            inserted = _flush_batch(collection, docs_batch)
            ingested += inserted
            logging.info("Inserted %s chunks (total %s)", inserted, ingested)

    inserted = _flush_batch(collection, docs_batch)
    if inserted:
        ingested += inserted
        logging.info("Inserted final %s chunks", inserted)

    if ingested == 0:
        logging.info("No data to ingest")
        return 0

    collection.flush()
    logging.info(
        "Ingested %s chunks into '%s' at %s (skipped %s files)",
        ingested,
        collection_name,
        zvec_uri,
        skipped,
    )
    return 0
