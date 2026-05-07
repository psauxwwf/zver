# Zver Agent Notes

- `main.py` is the only real entrypoint. It wires three flows: `embed` builds the local vector store, `query` reads it, and `mcp` serves the same search API over MCP.
- Repo layout is functional, not package-heavy: `embed/` owns ingestion/index creation, `search/` owns querying and MCP, `main.py` owns CLI routing and collection selection.

## Setup

- Install deps with `uv sync`.
- The working corpus and indexes are local state, not repo content: `/docs` is the source document tree, `/data/models` is the SentenceTransformer cache, `/data/zvec` is the persisted zvec store, `/bin` holds built binaries.

## Commands

- Ingest documents: `uv run python main.py embed --docs docs`
- Run a one-off query: `uv run python main.py query --docs docs --mode find_by_text_dense "your query"`
- Text search modes: `find_by_text_dense`, `find_by_text_bm25`, `find_by_text_hybrid`, `find_by_text_like`
- Document-name search modes: `all_by_name_dense`, `all_by_name_like`
- Start the MCP server: `uv run python main.py mcp --docs docs`
- Build the standalone binary: `task build:python` (writes `bin/zver` via Nuitka)
- Build in Docker: `task build:docker`
- Publish a release artifact: `task release` (uploads `bin/zver` to `devil666face/zver`)

## Gotchas

- Collection names are derived from the resolved `--docs` path. Use the same `--docs` value for `embed`, `query`, and `mcp` or searches will miss the intended collection.
- `query` has one special fallback: if `--docs` is left as the default `docs` and `data/zvec` contains exactly one collection, it will use that collection. Do not rely on this when multiple collections exist.
- Search naming is semantic, not implementation-shaped: `dense` = vector semantic search, `bm25` = lexical ranked search, `hybrid` = dense + bm25, `like` = literal substring filter.
- `bm25` and `hybrid` require collections built by the current embed pipeline with the sparse text field present. Older collections need re-embedding.
- `opencode.jsonc` expects the MCP server at `http://127.0.0.1:8000/mcp`; start `main.py mcp` before using the local MCP tool.

## Verification

- After each Python source change, run `task check:python`.
- Run the query smoke test against the current local index with `task test`.
- There is no repo-defined lint or typecheck suite beyond the compile check.
