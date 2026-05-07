from __future__ import annotations

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    path: str = Field(description="Original source file path")
    name: str = Field(description="Original file name with extension")
    stem: str = Field(description="File name without extension")
    parent: str = Field(description="Parent directory path")
    suffix: str = Field(description="File extension including the leading dot")
    size_bytes: int = Field(description="Source file size in bytes")
    mtime_ns: int = Field(description="Source file modification time in nanoseconds")
    chunk_index: int = Field(description="Zero-based chunk index inside the source file")
    chunk_count: int = Field(
        description="Total number of chunks produced for the source file"
    )


class SearchResult(BaseModel):
    id: str = Field(description="Stable chunk identifier")
    name: str = Field(
        description="Normalized source file name used for name-based search"
    )
    text: str = Field(description="Chunk text content")
    metadata: SourceMetadata = Field(description="Chunk and source metadata")
    score: float = Field(
        description="Match score; semantic similarity for embedding search or 1.0 for exact filter matches"
    )


def search_results_to_jsonable(results: list[SearchResult]) -> list[dict]:
    return [result.model_dump(mode="json") for result in results]
