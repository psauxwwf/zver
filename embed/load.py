from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from sentence_transformers import SentenceTransformer
from zvec import BM25EmbeddingFunction, Collection, Doc

from .document import (
    build_docs,
    normalize_source_name,
)
from .model import build_sentence_transformer, resolve_transformer_device
from .source import (
    build_hybrid_chunker,
    get_chunks,
    get_source_files,
)
from .store import (
    BM25_ENCODER_FILENAME,
    METADATA_FIELD,
    get_or_create_collection,
    read_doc_manifest,
    write_doc_manifest,
)


def _flush_batch(collection: Collection, docs_batch: list[Doc]) -> int:
    if not docs_batch:
        return 0

    collection.upsert(docs_batch)
    inserted = len(docs_batch)
    docs_batch.clear()
    return inserted


def _configure_model_storage(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(models_dir)


def _build_document_converter(device: str | None) -> DocumentConverter:
    if device != "cpu":
        return DocumentConverter()

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )


def _build_transformer(
    embed_model_name: str,
    models_dir: Path,
    device: str | None,
) -> tuple[SentenceTransformer, int]:
    transformer = build_sentence_transformer(embed_model_name, models_dir, device)
    dim = transformer.get_embedding_dimension()
    if dim is None:
        raise RuntimeError(
            f"Embedding dimension is undefined for model {embed_model_name}"
        )
    return transformer, dim


def _get_chunks(
    path: Path,
    converter: DocumentConverter,
    chunker,
    by_source: bool,
    chunk_min_chars: int,
) -> list[str]:
    return get_chunks(path, converter, chunker, by_source, chunk_min_chars)


def _build_docs_for_path(
    path: Path,
    chunks: list[str],
    transformer: SentenceTransformer,
    bm25_document_encoder: BM25EmbeddingFunction,
) -> list[Doc]:
    text_embeddings = transformer.encode(chunks).tolist()
    text_sparse_embeddings = [bm25_document_encoder.embed(chunk) for chunk in chunks]
    name = normalize_source_name(path)
    name_embedding = transformer.encode([name]).tolist()[0]
    return build_docs(
        path,
        chunks,
        text_embeddings,
        text_sparse_embeddings,
        name_embedding,
    )


def _path_filter(path: Path) -> str:
    path_fragment = f'"path": {json.dumps(str(path), ensure_ascii=False)}'
    return (
        f"{METADATA_FIELD} LIKE {json.dumps(f'%{path_fragment}%', ensure_ascii=False)}"
    )


def _write_bm25_encoder(
    zvec_uri: Path,
    collection_name: str,
    bm25_document_encoder: BM25EmbeddingFunction,
) -> None:
    encoder = getattr(bm25_document_encoder, "_encoder", None)
    if encoder is None:
        raise RuntimeError("BM25 encoder is unavailable for persistence")

    encoder_path = zvec_uri / collection_name / BM25_ENCODER_FILENAME
    encoder.dump(str(encoder_path))


def _hash_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_existing_chunks(
    collection: Collection, doc_ids: list[str]
) -> list[str] | None:
    if not doc_ids:
        return []

    docs_by_id = collection.fetch(doc_ids)
    chunks: list[str] = []
    for doc_id in doc_ids:
        doc = docs_by_id.get(doc_id)
        if doc is None:
            return None

        text = doc.field("text")
        if not isinstance(text, str):
            return None
        chunks.append(text)

    return chunks


def load_documents(
    docs_dir: Path,
    collection_name: str,
    embed_model_name: str,
    zvec_uri: Path,
    models_dir: Path,
    batch_size: int,
    chunk_min_chars: int,
    by_source: bool,
    include_ext: list[str] | None,
) -> int:
    device = resolve_transformer_device()
    _configure_model_storage(models_dir)
    converter = _build_document_converter(device)
    transformer, dim = _build_transformer(embed_model_name, models_dir, device)
    chunker = build_hybrid_chunker(transformer)
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
    prepared_docs: list[tuple[Path, str, list[str]]] = []
    empty_paths: list[tuple[Path, str]] = []
    preserved_chunks: list[str] = []
    doc_manifest = (
        read_doc_manifest(collection_name=collection_name, zvec_uri=zvec_uri) or {}
    )

    for path in files:
        file_hash = _hash_file(path)
        manifest_entry = doc_manifest.get(str(path))
        if manifest_entry is not None and manifest_entry["hash"] == file_hash:
            existing_chunks = _load_existing_chunks(collection, manifest_entry["ids"])
            if existing_chunks is not None:
                skipped += 1
                preserved_chunks.extend(existing_chunks)
                logging.info("Skipped %s (unchanged)", path)
                continue

            logging.warning(
                "Reprocessing %s because manifest ids are missing from collection",
                path,
            )

        try:
            chunks = _get_chunks(
                path=path,
                converter=converter,
                chunker=chunker,
                by_source=by_source,
                chunk_min_chars=chunk_min_chars,
            )
        except Exception as exc:  # pragma: no cover - logging only
            skipped += 1
            logging.warning("Skipping %s: %s", path, exc)
            continue

        logging.info(
            "Chunked %s into %s chunks with lengths=%s",
            path,
            len(chunks),
            [len(chunk) for chunk in chunks],
        )

        if not chunks:
            empty_paths.append((path, file_hash))
            logging.debug("No chunks produced for %s", path)
            continue

        prepared_docs.append((path, file_hash, chunks))

    text_corpus = preserved_chunks + [
        chunk for _, _, chunks in prepared_docs for chunk in chunks
    ]
    if not text_corpus:
        for path, file_hash in empty_paths:
            collection.delete_by_filter(_path_filter(path))
            doc_manifest[str(path)] = {"hash": file_hash, "ids": []}
        write_doc_manifest(
            collection_name=collection_name,
            zvec_uri=zvec_uri,
            manifest=doc_manifest,
        )
        logging.info("No data to ingest")
        return 0

    bm25_document_encoder = BM25EmbeddingFunction(
        corpus=text_corpus,
        encoding_type="document",
    )
    _write_bm25_encoder(
        zvec_uri=zvec_uri,
        collection_name=collection_name,
        bm25_document_encoder=bm25_document_encoder,
    )

    for path, file_hash in empty_paths:
        collection.delete_by_filter(_path_filter(path))
        doc_manifest[str(path)] = {"hash": file_hash, "ids": []}

    for path, file_hash, chunks in prepared_docs:
        docs_for_path = _build_docs_for_path(
            path,
            chunks,
            transformer,
            bm25_document_encoder,
        )
        doc_manifest[str(path)] = {
            "hash": file_hash,
            "ids": [doc.id for doc in docs_for_path],
        }

        if (
            batch_size > 0
            and docs_batch
            and len(docs_batch) + len(docs_for_path) > batch_size
        ):
            inserted = _flush_batch(collection, docs_batch)
            ingested += inserted
            logging.info("Upserted %s chunks (total %s)", inserted, ingested)

        collection.delete_by_filter(_path_filter(path))
        docs_batch.extend(docs_for_path)

        if batch_size > 0 and len(docs_batch) >= batch_size:
            inserted = _flush_batch(collection, docs_batch)
            ingested += inserted
            logging.info("Upserted %s chunks (total %s)", inserted, ingested)

    inserted = _flush_batch(collection, docs_batch)
    if inserted:
        ingested += inserted
        logging.info("Upserted final %s chunks", inserted)

    if ingested == 0:
        logging.info("No data to ingest")
        return 0

    collection.flush()
    write_doc_manifest(
        collection_name=collection_name,
        zvec_uri=zvec_uri,
        manifest=doc_manifest,
    )
    logging.info(
        "Ingested %s chunks into '%s' at %s (skipped %s files)",
        ingested,
        collection_name,
        zvec_uri,
        skipped,
    )
    return 0
