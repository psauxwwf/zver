from __future__ import annotations

import logging
import warnings
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def _is_local_cache_miss(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "local_files_only",
            "outgoing traffic has been disabled",
            "couldn't connect",
            "cannot find the requested files in the local cache",
            "not the path to a directory containing a file named",
        )
    )


def _resolve_transformer_device() -> str | None:
    if not torch.cuda.is_available():
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Found GPU0 .* compute capability",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"\n?NVIDIA .* is not compatible with the current PyTorch installation\.",
                category=UserWarning,
            )
            major, minor = torch.cuda.get_device_capability()
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.warning("Unable to inspect CUDA device, using CPU: %s", exc)
        return "cpu"

    device_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if device_arch in supported_arches:
        return None

    logging.warning(
        "CUDA device %s (%s) is unsupported by this PyTorch build; using CPU instead",
        device_name,
        device_arch,
    )
    return "cpu"


def build_sentence_transformer(
    embed_model_name: str,
    models_dir: Path,
) -> SentenceTransformer:
    device = _resolve_transformer_device()
    kwargs: dict[str, str | bool] = {
        "cache_folder": str(models_dir),
        "local_files_only": True,
    }
    if device is not None:
        kwargs["device"] = device

    try:
        return SentenceTransformer(embed_model_name, **kwargs)
    except Exception as exc:
        if not _is_local_cache_miss(exc):
            raise

    logging.info(
        "Model %s is not fully cached; checking Hugging Face", embed_model_name
    )
    kwargs["local_files_only"] = False
    return SentenceTransformer(embed_model_name, **kwargs)
