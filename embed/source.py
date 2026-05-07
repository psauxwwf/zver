from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from sentence_transformers import SentenceTransformer


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


def build_hybrid_chunker(transformer: SentenceTransformer) -> HybridChunker:
    tokenizer = HuggingFaceTokenizer(tokenizer=transformer.tokenizer)
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def get_chunks(
    path: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
    by_source: bool,
    chunk_min_chars: int,
) -> list[str]:
    chunks = convert_file(path, converter, chunker)

    if by_source:
        full_text = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
        if not full_text:
            return []
        return [full_text]

    return merge_chunks(chunks, chunk_min_chars)


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
