from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .search import (
    MAX_QUERY_TOP_K,
    SearchContext,
    all_by_name,
    all_by_name_embed,
    all_names,
    find_by_text,
    find_by_text_embed,
)

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
        description="Search chunks by semantic similarity over text embeddings.",
        structured_output=True,
    )
    def find_by_text_embed_tool(
        query: str,
        top_k: int = 5,
        nprobe: int = 10,
    ) -> list[dict]:
        return find_by_text_embed(ctx=ctx, query=query, top_k=top_k, nprobe=nprobe)

    @mcp.tool(
        name="find_by_text",
        title="Find By Text",
        description="Search chunks by plain text LIKE filter.",
        structured_output=True,
    )
    def find_by_text_tool(
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        return find_by_text(ctx=ctx, query=query, top_k=top_k)

    @mcp.tool(
        name="all_by_name",
        title="All By Name",
        description="Return all chunks whose file name matches the query.",
        structured_output=True,
    )
    def all_by_name_tool(query: str) -> list[dict]:
        return all_by_name(ctx=ctx, query=query)

    @mcp.tool(
        name="all_by_name_embed",
        title="All By Name Embed",
        description="Return all chunks for file names matched semantically by embedding.",
        structured_output=True,
    )
    def all_by_name_embed_tool(
        query: str,
        top_k: int = 5,
        nprobe: int = 10,
    ) -> list[dict]:
        return all_by_name_embed(ctx=ctx, query=query, top_k=top_k, nprobe=nprobe)

    @mcp.tool(
        name="all_names",
        title="All Names",
        description=(
            "Return all distinct file names stored in the collection, up to the current "
            f"query limit of {MAX_QUERY_TOP_K} documents."
        ),
        structured_output=True,
    )
    def all_names_tool() -> list[str]:
        return all_names(ctx)

    return mcp
