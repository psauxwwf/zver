# Zver Agent Notes

- `main.py` is the only real entrypoint. It wires three flows: `embed` builds the local vector store, `query` reads it, and `mcp` serves the same search API over MCP.
- Repo layout is functional, not package-heavy: `embed/` owns ingestion/index creation, `search/` owns querying and MCP, `main.py` owns CLI routing and collection selection.

## Setup

- Install deps with `uv sync`.
- The working corpus and indexes are local state, not repo content: `/docs` is the source document tree, `/data/models` is the SentenceTransformer cache, `/data/zvec` is the persisted zvec store, `/bin` holds built binaries.

## Commands

- Ingest documents: `uv run python main.py embed --docs docs`
- Run a one-off query: `uv run python main.py query --docs docs --mode find_by_text_embed "your query"`
- Start the MCP server: `uv run python main.py mcp --docs docs`
- Build the standalone binary: `task build:python` (writes `bin/zver` via Nuitka)
- Build in Docker: `task build:docker`
- Publish a release artifact: `task release` (uploads `bin/zver` to `devil666face/zver`)

## Gotchas

- Collection names are derived from the resolved `--docs` path. Use the same `--docs` value for `embed`, `query`, and `mcp` or searches will miss the intended collection.
- `query` has one special fallback: if `--docs` is left as the default `docs` and `data/zvec` contains exactly one collection, it will use that collection. Do not rely on this when multiple collections exist.
- `opencode.jsonc` expects the MCP server at `http://127.0.0.1:8000/mcp`; start `main.py mcp` before using the local MCP tool.

## Verification

- After each Python source change, run `python -m py_compile main.py embed/*.py search/*.py`.
- There is no repo-defined test, lint, or typecheck suite beyond that compile check.
