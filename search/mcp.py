from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .search import (
    MAX_QUERY_TOP_K,
    SearchContext,
    all_by_name,
    all_by_name_embed,
    all_names,
    find_by_text,
    find_by_text_embed,
)
from .types import SearchResult

MCP_HOST = "127.0.0.1"
MCP_PORT = 8000
MCP_PATH = "/mcp"
def build_mcp_server(ctx: SearchContext) -> FastMCP:
    mcp = FastMCP(
        "zver",
        json_response=True,
        host=MCP_HOST,
        port=MCP_PORT,
        streamable_http_path=MCP_PATH,
    )

    @mcp.tool(
        name="find_by_text_embed",
        title="Find By Text Embed",
        description=(
            "Use for semantic content search when the query describes meaning, topic, or intent "
            "and exact wording may differ from the source text. Input: query string, optional "
            "top_k and nprobe. Output: list of matching document chunks with fields id, name, "
            "text, metadata, and similarity score."
        ),
        structured_output=True,
    )
    def find_by_text_embed_tool(
        query: Annotated[
            str,
            Field(
                description="Natural-language query for semantic search in document content. Use topics, questions, paraphrases, or intent descriptions. Empty string returns an empty list."
            ),
        ],
        top_k: Annotated[
            int,
            Field(
                description="Maximum number of matching chunks to return. Higher values return more candidates."
            ),
        ] = 5,
        nprobe: Annotated[
            int,
            Field(
                description="IVF vector-search breadth. Higher values may improve recall at the cost of more work."
            ),
        ] = 10,
    ) -> list[SearchResult]:
        return find_by_text_embed(ctx=ctx, query=query, top_k=top_k, nprobe=nprobe)

    @mcp.tool(
        name="find_by_text",
        title="Find By Text",
        description=(
            "Use for exact or near-exact substring search when you expect the source text to "
            "contain the same words, codes, names, or phrases. Input: query string and optional "
            "top_k. Output: list of matching document chunks with fields id, name, text, metadata, "
            "and score=1.0."
        ),
        structured_output=True,
    )
    def find_by_text_tool(
        query: Annotated[
            str,
            Field(
                description="Exact text fragment to find inside chunk text. Best for literal phrases, identifiers, codes, or known wording. Empty string returns an empty list."
            ),
        ],
        top_k: Annotated[
            int,
            Field(description="Maximum number of matching chunks to return."),
        ] = 5,
    ) -> list[SearchResult]:
        return find_by_text(ctx=ctx, query=query, top_k=top_k)

    @mcp.tool(
        name="all_by_name",
        title="All By Name",
        description=(
            "Use when the user knows all or part of the file name and wants the full content for "
            "every matching source. Input: file-name query string. Output: all chunks for files "
            "whose normalized name contains that substring, each with id, name, text, metadata, "
            "and score=1.0."
        ),
        structured_output=True,
    )
    def all_by_name_tool(
        query: Annotated[
            str,
            Field(
                description="Exact substring to match against normalized file names. Use when the document name or part of it is already known. Empty string returns an empty list."
            ),
        ],
    ) -> list[SearchResult]:
        return all_by_name(ctx=ctx, query=query)

    @mcp.tool(
        name="all_by_name_embed",
        title="All By Name Embed",
        description=(
            "Use when the user refers to a document by approximate meaning or paraphrased title and "
            "you need the full contents of the most relevant files. Input: query string, optional "
            "top_k and nprobe. Output: all chunks for semantically matched file names, each with "
            "id, name, text, metadata, and a file-level similarity score."
        ),
        structured_output=True,
    )
    def all_by_name_embed_tool(
        query: Annotated[
            str,
            Field(
                description="Natural-language description of the desired document name or title. Use when the exact file name is unknown. Empty string returns an empty list."
            ),
        ],
        top_k: Annotated[
            int,
            Field(
                description="Maximum number of candidate file names to keep before returning all chunks for those files."
            ),
        ] = 5,
        nprobe: Annotated[
            int,
            Field(
                description="IVF vector-search breadth for semantic file-name matching. Higher values may improve recall."
            ),
        ] = 10,
    ) -> list[SearchResult]:
        return all_by_name_embed(ctx=ctx, query=query, top_k=top_k, nprobe=nprobe)

    @mcp.tool(
        name="all_names",
        title="All Names",
        description=(
            "Use to inspect what documents are available before choosing a more specific tool. "
            "Input: none. Output: sorted list of distinct normalized file names currently stored in "
            f"the collection, limited by the current backend query cap of {MAX_QUERY_TOP_K} documents."
        ),
        structured_output=True,
    )
    def all_names_tool() -> list[str]:
        return all_names(ctx)

    return mcp
