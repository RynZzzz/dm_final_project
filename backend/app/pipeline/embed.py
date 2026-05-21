"""Sentence-transformer embeddings. Single model, lazy-loaded, used for
embedding concepts and comments for similarity matching."""

import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

_embedder = None
_embedder_device: str | None = None


def get_embedder():
    """Return a cached SentenceTransformer model.

    Auto-detects CUDA with CPU fallback. Loads on first call.
    """
    global _embedder, _embedder_device

    if _embedder is not None:
        return _embedder

    import torch
    from sentence_transformers import SentenceTransformer

    _embedder_device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if _embedder_device == "cuda" else torch.float32

    logger.info("Loading embedder model '%s' on %s (%s)", _EMBED_MODEL, _embedder_device, dtype)
    _embedder = SentenceTransformer(_EMBED_MODEL, device=_embedder_device, model_kwargs={"torch_dtype": dtype})
    logger.info("Embedder model ready on %s", _embedder_device)

    return _embedder


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Batch-encode a list of strings. Returns a (N, 768) float32 array.

    Embeddings are L2-normalised, so dot product equals cosine similarity.
    Empty or whitespace-only strings are replaced with a single space to
    avoid NaN outputs from the model.
    """
    embedder = get_embedder()
    sanitized = [t if t and t.strip() else " " for t in texts]

    embeddings = embedder.encode(
        sanitized,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)
