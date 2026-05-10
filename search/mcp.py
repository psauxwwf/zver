from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from .search import (
    SearchContext,
    all_by_name_dense,
    all_by_name_grep,
    all_names,
    find_by_text_bm25,
    find_by_text_dense,
    find_by_text_hybrid,
    find_by_text_grep,
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
        name="find_by_text_dense",
        title="Find By Text Dense",
        description=(
            "Use for semantic content search when the query describes meaning, topic, or intent "
            "and exact wording may differ from the source text. Input: query string, optional "
            "top_k and nprobe. Output: list of matching document chunks with fields id, name, "
            "text, metadata, and similarity score. Best when the user is asking by meaning rather "
            "than exact terms, but hybrid is the recommended default search tool."
        ),
        structured_output=True,
    )
    def find_by_text_dense_tool(
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
        return find_by_text_dense(ctx=ctx, query=query, top_k=top_k, nprobe=nprobe)

    @mcp.tool(
        name="find_by_text_bm25",
        title="Find By Text BM25",
        description=(
            "Use for lexical ranked search when exact query terms matter and you want BM25-style "
            "relevance instead of a raw substring filter. Input: query string and optional top_k. "
            "Output: list of matching document chunks with fields id, name, text, metadata, and score. "
            "Best for jargon, acronyms, commands, and rare terms."
        ),
        structured_output=True,
    )
    def find_by_text_bm25_tool(
        query: Annotated[
            str,
            Field(
                description="Natural-language or keyword query for lexical BM25 search in document content. Empty string returns an empty list."
            ),
        ],
        top_k: Annotated[
            int,
            Field(description="Maximum number of matching chunks to return."),
        ] = 5,
    ) -> list[SearchResult]:
        return find_by_text_bm25(ctx=ctx, query=query, top_k=top_k)

    @mcp.tool(
        name="find_by_text_hybrid",
        title="Find By Text Hybrid",
        description=(
            "Use for combined semantic and lexical retrieval over document content. It mixes dense "
            "vector search with BM25-style sparse search and re-ranks the merged candidates. Input: "
            "query string, optional top_k and nprobe. Output: list of matching document chunks with "
            "fields id, name, text, metadata, and score. Recommended default search tool when both "
            "meaning and exact terms may matter."
        ),
        structured_output=True,
    )
    def find_by_text_hybrid_tool(
        query: Annotated[
            str,
            Field(
                description="Natural-language query for hybrid dense plus BM25 search in document content. Empty string returns an empty list."
            ),
        ],
        top_k: Annotated[
            int,
            Field(
                description="Maximum number of matching chunks to return after re-ranking."
            ),
        ] = 5,
        nprobe: Annotated[
            int,
            Field(
                description="IVF vector-search breadth for the dense branch. Higher values may improve recall at the cost of more work."
            ),
        ] = 10,
    ) -> list[SearchResult]:
        return find_by_text_hybrid(ctx=ctx, query=query, top_k=top_k, nprobe=nprobe)

    @mcp.tool(
        name="find_by_text_grep",
        title="Find By Text Grep",
        description=(
            "Use for GNU grep pattern search when you expect the source text to match a grep regular "
            "expression. This uses ordinary grep syntax, not LIKE and not CLI flags. Input: query "
            "string and optional top_k. Output: list of matching document chunks with fields id, "
            "name, text, metadata, and score=1.0. Example patterns: 'evilginx', '^Chapter [0-9][0-9]*$', "
            "'mysql_.*', '\\(error\\|warning\\)'."
        ),
        structured_output=True,
    )
    def find_by_text_grep_tool(
        query: Annotated[
            str,
            Field(
                description="GNU grep pattern to match against chunk text. Use ordinary grep syntax such as 'evilginx', '^Step [0-9][0-9]*$', 'kesl-control --start-task [0-9][0-9]*', or '\\(error\\|warning\\)'. Empty string returns an empty list."
            ),
        ],
        top_k: Annotated[
            int,
            Field(description="Maximum number of matching chunks to return."),
        ] = 5,
    ) -> list[SearchResult]:
        return find_by_text_grep(ctx=ctx, query=query, top_k=top_k)

    @mcp.tool(
        name="all_by_name_grep",
        title="All By Name Grep",
        description=(
            "Use when the user knows all or part of the file name and wants the full content for "
            "every matching source. Input: file-name grep pattern using ordinary GNU grep syntax. "
            "Output: all chunks for files whose normalized name matches that grep pattern, each with "
            "id, name, text, metadata, and score=1.0. Example patterns: 'KESL', '^Project .* Guide$', "
            "'KSC_[0-9][0-9][0-9][0-9]'."
        ),
        structured_output=True,
    )
    def all_by_name_grep_tool(
        query: Annotated[
            str,
            Field(
                description="GNU grep pattern to match against normalized file names. Use ordinary grep syntax such as 'KESL', '^Project .* Guide$', or 'Formulary\\|Guide'. Empty string returns an empty list."
            ),
        ],
    ) -> list[SearchResult]:
        return all_by_name_grep(ctx=ctx, query=query)

    @mcp.tool(
        name="all_by_name_dense",
        title="All By Name Dense",
        description=(
            "Use when the user refers to a document by approximate meaning or paraphrased title and "
            "you need the full contents of the most relevant files. Input: query string, optional "
            "top_k and nprobe. Output: all chunks for semantically matched file names, each with "
            "id, name, text, metadata, and a file-level similarity score. Best when the document "
            "name is unknown but its meaning is known."
        ),
        structured_output=True,
    )
    def all_by_name_dense_tool(
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
        return all_by_name_dense(ctx=ctx, query=query, top_k=top_k, nprobe=nprobe)

    @mcp.tool(
        name="all_names",
        title="All Names",
        description=(
            "Use to inspect what documents are available before choosing a more specific tool. "
            "Input: none. Output: sorted list of distinct normalized file names currently stored in "
            "the collection."
        ),
        structured_output=True,
    )
    def all_names_tool() -> list[str]:
        return all_names(ctx)

    return mcp
