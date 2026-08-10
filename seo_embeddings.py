"""Offline local embedding-model loading and efficient query encoding."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# The workflow must never silently download a model during a long job.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR.parent / "models" / "multilingual-e5-small"
LEGACY_MODEL_NAMES = {"intfloat/multilingual-e5-small", "local:multilingual-e5-small"}
REQUIRED_MODEL_FILES = ("config.json", "modules.json", "model.safetensors", "tokenizer.json")


def resolve_local_model(value: str | None = None) -> Path:
    """Return the supported local model directory and validate it early."""
    raw = str(value or "local:multilingual-e5-small").strip()
    candidate = DEFAULT_MODEL_DIR if raw in LEGACY_MODEL_NAMES else Path(raw)
    if not candidate.is_absolute():
        candidate = (PROJECT_DIR / candidate).resolve()
    missing = [name for name in REQUIRED_MODEL_FILES if not (candidate / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Local embedding model is incomplete: "
            f"{candidate}. Missing: {', '.join(missing)}. "
            "Place multilingual-e5-small in ../models/multilingual-e5-small."
        )
    return candidate


def choose_device(requested: str | None = None) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("embedding_device must be auto, cuda, or cpu")
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_local_embedding_model(config: dict[str, Any]) -> tuple[Any, str, Path]:
    """Load E5 from disk only and report the actual compute device."""
    from sentence_transformers import SentenceTransformer

    model_dir = resolve_local_model(str(config.get("embedding_model", "local:multilingual-e5-small")))
    device = choose_device(str(config.get("embedding_device", "auto")))
    return SentenceTransformer(str(model_dir), device=device, local_files_only=True), device, model_dir


def encode_queries(
    model: Any,
    texts: Any,
    batch_size: int,
    prefix: str = "query: ",
) -> np.ndarray:
    """Encode query strings with the E5 query prefix and normalized vectors."""
    values = [f"{prefix}{str(text)}" for text in texts]
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(
        model.encode(
            values,
            batch_size=max(1, int(batch_size)),
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
