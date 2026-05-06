from pathlib import Path

DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DEFAULT_MILVUS_URI = Path("data") / "zvec"
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
