from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_MODELS_DIR = Path("data") / "models"
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DEFAULT_ZVEC_URI = Path("data") / "zvec"
DEFAULT_TOP_K = 5
DEFAULT_NPROBE = 10
DEFAULT_BATCH_SIZE = 1024
DEFAULT_CHUNK_MIN_CHARS = 256
DEFAULT_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
    ".md",
    ".rst",
)

MODE_FIND_BY_TEXT_EMBED = "find_by_text_embed"
MODE_FIND_BY_TEXT = "find_by_text"
MODE_ALL_BY_NAME = "all_by_name"
MODE_ALL_BY_NAME_EMBED = "all_by_name_embed"


def _batch_size_arg(value: str) -> int:
    batch_size = int(value)
    if batch_size < 1 or batch_size > 1024:
        raise argparse.ArgumentTypeError("batch size must be in range 1..1024")
    return batch_size


def _add_docs_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--docs",
        type=str,
        default=str(DEFAULT_DOCS_DIR),
        help="Directory with documents to ingest",
    )


def _add_embed_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL_NAME,
        help="Sentence-Transformers embedding model",
    )


def _add_zvec_uri_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--zvec-uri",
        type=str,
        default=str(DEFAULT_ZVEC_URI),
        help="Directory where zvec data is stored",
    )


def _add_log_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into zvec and run searches against them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser(
        "embed",
        help="Ingest documents into zvec",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_docs_arg(embed_parser)
    _add_embed_model_arg(embed_parser)
    _add_zvec_uri_arg(embed_parser)
    embed_parser.add_argument(
        "--batch-size",
        type=_batch_size_arg,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks to upsert per batch",
    )
    embed_parser.add_argument(
        "--chunk-min-chars",
        dest="chunk_min_chars",
        type=int,
        default=DEFAULT_CHUNK_MIN_CHARS,
        help="Target minimum chunk size in characters (0 - disables merging)",
    )
    embed_parser.add_argument(
        "--by-source",
        action="store_true",
        help="Merge each source into one chunk before embedding",
    )
    embed_parser.add_argument(
        "--ext",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        metavar=".EXT",
        help="File extensions to ingest, case-insensitive",
    )
    _add_log_arg(embed_parser)

    query_parser = subparsers.add_parser(
        "query",
        help="Run one-off search query",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_docs_arg(query_parser)
    _add_embed_model_arg(query_parser)
    _add_zvec_uri_arg(query_parser)
    query_parser.add_argument(
        "--mode",
        default=MODE_FIND_BY_TEXT_EMBED,
        choices=[
            MODE_FIND_BY_TEXT_EMBED,
            MODE_FIND_BY_TEXT,
            MODE_ALL_BY_NAME,
            MODE_ALL_BY_NAME_EMBED,
        ],
        help="Search mode",
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Maximum number of results to return",
    )
    query_parser.add_argument(
        "--nprobe",
        type=int,
        default=DEFAULT_NPROBE,
        help="IVF nprobe value for vector search",
    )
    query_parser.add_argument(
        "--list",
        action="store_true",
        help="List all distinct file names",
    )
    query_parser.add_argument(
        "query",
        nargs="?",
        default="",
        metavar="QUERY",
        help="Query text",
    )
    _add_log_arg(query_parser)

    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Run MCP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_docs_arg(mcp_parser)
    _add_embed_model_arg(mcp_parser)
    _add_zvec_uri_arg(mcp_parser)
    _add_log_arg(mcp_parser)

    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.command == "embed" and args.chunk_min_chars < 0:
        raise ValueError("chunk min chars must be greater than or equal to 0")

    return args
