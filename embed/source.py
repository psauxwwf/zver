from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker


def get_source_files(
    root: Path,
    include_ext: Iterable[str] | None = None,
) -> Iterable[Path]:
    include_lower = {ext.lower() for ext in include_ext} if include_ext else None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if include_lower and path.suffix.lower() not in include_lower:
            logging.debug("Skipping %s (extension not allowed)", path)
            continue
        yield path


def convert_file(
    path: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
) -> list[str]:
    doc = converter.convert(source=str(path)).document
    chunk_iter = chunker.chunk(dl_doc=doc)
    return [chunker.contextualize(chunk=chunk) for chunk in chunk_iter]


def merge_chunks(
    chunks: list[str],
    chunk_min_chars: int,
) -> list[str]:
    clean_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    result: list[str] = []

    def _flush(buffer_parts: list[str]) -> None:
        if buffer_parts:
            result.append("\n\n".join(buffer_parts))

    if chunk_min_chars <= 0:
        return clean_chunks

    buffer_parts: list[str] = []
    buffer_len = 0

    for chunk in clean_chunks:
        chunk_len = len(chunk)
        next_len = buffer_len + 2 + chunk_len

        if not buffer_parts:
            buffer_parts = [chunk]
            buffer_len = chunk_len
            continue

        if buffer_len >= chunk_min_chars:
            _flush(buffer_parts)
            buffer_parts = [chunk]
            buffer_len = chunk_len
        else:
            buffer_parts.append(chunk)
            buffer_len = next_len

    _flush(buffer_parts)

    return result
