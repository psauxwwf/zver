from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_MODELS_DIR = Path("data") / "models"
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DEFAULT_ZVEC_URI = Path("data") / "zvec"
DEFAULT_TOP_K = 5
DEFAULT_NPROBE = 10
DEFAULT_BATCH_SIZE = 512
DEFAULT_CHUNK_CHARS = 1024
DEFAULT_MAX_TEXT_LEN = 8192
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert documents to embeddings and store them in zvec",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=str(DEFAULT_DOCS_DIR),
        help="Directory with source documents",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL_NAME,
        help="Sentence-Transformers model to use for embeddings",
    )
    parser.add_argument(
        "--query",
        nargs="?",
        default=None,
        const="",
        metavar="TEXT",
        help="Ad-hoc query to run and print as JSON",
    )
    parser.add_argument(
        "--mode",
        default=MODE_FIND_BY_TEXT_EMBED,
        choices=[
            MODE_FIND_BY_TEXT_EMBED,
            MODE_FIND_BY_TEXT,
            MODE_ALL_BY_NAME,
            MODE_ALL_BY_NAME_EMBED,
        ],
        help="Search mode for --query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of search results or candidate names to use",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=DEFAULT_NPROBE,
        help="IVF nprobe search parameter for vector search",
    )
    parser.add_argument(
        "--zvec-uri",
        type=str,
        default=str(DEFAULT_ZVEC_URI),
        help="Directory where zvec collections are stored",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks per zvec insert (0 = all at once)",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=DEFAULT_CHUNK_CHARS,
        help="Approximate min characters per chunk (0 disables merging)",
    )
    parser.add_argument(
        "--max-text-len",
        type=int,
        default=DEFAULT_MAX_TEXT_LEN,
        help="Hard max characters per stored chunk",
    )
    parser.add_argument(
        "--by-source",
        action="store_true",
        help=(
            "Merge all chunks for a single source into one text and "
            "re-split it into pieces of up to --max-text-len characters"
        ),
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        metavar=".EXT",
        help="File extensions to ingest (case-insensitive)",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_text_len < 1:
        args.max_text_len = DEFAULT_MAX_TEXT_LEN

    if args.chunk_chars > 0 and args.chunk_chars > args.max_text_len:
        args.chunk_chars = args.max_text_len

    return args
