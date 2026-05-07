from __future__ import annotations

import hashlib
from pathlib import Path

from zvec import Doc

from .store import (
    METADATA_FIELD,
    NAME_EMBEDDING_FIELD,
    NAME_FIELD,
    TEXT_EMBEDDING_FIELD,
    TEXT_SPARSE_EMBEDDING_FIELD,
    TEXT_FIELD,
    serialize_metadata,
)


def normalize_source_name(path: Path) -> str:
    raw_name = path.stem
    return raw_name.strip() or raw_name


def build_metadata(
    path: Path,
    chunk_index: int,
    chunk_count: int,
) -> str:
    stat = path.stat()

    return serialize_metadata(
        {
            "path": str(path),
            "name": path.name,
            "stem": path.stem,
            "parent": str(path.parent),
            "suffix": path.suffix,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
        }
    )


def build_doc_id(path: Path, chunk_index: int) -> str:
    payload = f"{path.as_posix()}\0{chunk_index}".encode()
    return hashlib.sha256(payload).hexdigest()


def build_docs(
    path: Path,
    chunks: list[str],
    text_embeddings: list[list[float]],
    text_sparse_embeddings: list[dict[int, float]],
    name_embedding: list[float],
) -> list[Doc]:
    name = normalize_source_name(path)
    chunk_count = len(chunks)
    docs: list[Doc] = []

    for chunk_index, (chunk, text_embedding, text_sparse_embedding) in enumerate(
        zip(chunks, text_embeddings, text_sparse_embeddings, strict=True)
    ):
        docs.append(
            Doc(
                id=build_doc_id(path, chunk_index),
                fields={
                    NAME_FIELD: name,
                    TEXT_FIELD: chunk,
                    METADATA_FIELD: build_metadata(path, chunk_index, chunk_count),
                },
                vectors={
                    NAME_EMBEDDING_FIELD: name_embedding,
                    TEXT_EMBEDDING_FIELD: text_embedding,
                    TEXT_SPARSE_EMBEDDING_FIELD: text_sparse_embedding,
                },
            )
        )

    return docs
