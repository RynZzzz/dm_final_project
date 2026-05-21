"""Zero-shot classification of timestamped comments using a pretrained
DeBERTa-v3 NLI model. No training required."""

import logging
import time

from app import state
from app.config import settings
from app.models import ClassificationResult, ClassifiedComment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Candidate labels — phrasing matters for zero-shot quality
# ---------------------------------------------------------------------------

CANDIDATE_LABELS = [
    "expresses confusion or difficulty understanding the content",
    "asks a clarifying question about the content",
    "expresses frustration with the explanation",
    "gives positive feedback or appreciation",
    "is neutral or off-topic",
]

TROUBLE_LABELS = {
    "expresses confusion or difficulty understanding the content",
    "asks a clarifying question about the content",
    "expresses frustration with the explanation",
}

_NEUTRAL_LABEL = "is neutral or off-topic"
_NEUTRAL_SCORES = {lbl: 0.0 for lbl in CANDIDATE_LABELS} | {_NEUTRAL_LABEL: 1.0}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ClassificationError(Exception):
    pass

class VideoNotReadyError(ClassificationError):
    pass

class CommentsNotReadyError(ClassificationError):
    pass

class ModelLoadError(ClassificationError):
    pass


# ---------------------------------------------------------------------------
# Model loading — lazy, cached at module level
# ---------------------------------------------------------------------------

_classifier = None
_classifier_device: str | None = None


def get_classifier():
    """Return a cached zero-shot classification pipeline.

    Auto-detects CUDA with CPU fallback. Loads on first call.
    Raises ModelLoadError if the model cannot be loaded.
    """
    global _classifier, _classifier_device

    if _classifier is not None:
        return _classifier

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    _classifier_device = "cuda" if device == 0 else "cpu"
    dtype = torch.float16 if device == 0 else torch.float32

    logger.info(
        "Loading classifier model '%s' on %s (%s)",
        settings.CLASSIFIER_MODEL, _classifier_device, dtype,
    )
    try:
        _classifier = pipeline(
            "zero-shot-classification",
            model=settings.CLASSIFIER_MODEL,
            device=device,
            torch_dtype=dtype,
        )
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load classifier model '{settings.CLASSIFIER_MODEL}': {exc}"
        ) from exc

    logger.info("Classifier model ready on %s", _classifier_device)
    return _classifier


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_comments(
    comments: list[dict],
    threshold: float,
    batch_size: int = 32,
) -> list[ClassifiedComment]:
    """Classify a list of comment dicts (already filtered to timestamped ones).

    Comment dicts are expected to have at minimum: id, text, author,
    timestamps_seconds (list[int]).

    Returns ClassifiedComment objects in the same order as the input.
    Empty-text comments receive a synthetic neutral classification.
    """
    classifier = get_classifier()

    # Separate empty-text comments to avoid passing whitespace to the model
    valid_indices = [i for i, c in enumerate(comments) if c.get("text", "").strip()]
    empty_indices = set(range(len(comments))) - set(valid_indices)

    valid_texts = [comments[i]["text"] for i in valid_indices]

    all_outputs: list[dict] = []
    for start in range(0, len(valid_texts), batch_size):
        batch = valid_texts[start : start + batch_size]
        raw = classifier(batch, CANDIDATE_LABELS, multi_label=False)
        # HF returns a dict for single strings, list for multiple
        if isinstance(raw, dict):
            raw = [raw]
        all_outputs.extend(raw)
        logger.debug(
            "Classified batch %d–%d / %d",
            start + 1, start + len(batch), len(valid_texts),
        )

    results: list[ClassifiedComment | None] = [None] * len(comments)

    for local_idx, global_idx in enumerate(valid_indices):
        out = all_outputs[local_idx]
        top_label: str = out["labels"][0]
        top_score: float = float(out["scores"][0])
        all_scores = {
            lbl: float(scr)
            for lbl, scr in zip(out["labels"], out["scores"])
        }
        c = comments[global_idx]
        results[global_idx] = ClassifiedComment(
            comment_id=c.get("id", ""),
            author=c.get("author", ""),
            text=c.get("text", ""),
            timestamps_in_text=c.get("timestamps_seconds", []),
            classification=ClassificationResult(
                predicted_label=top_label,
                confidence=round(top_score, 4),
                all_scores={k: round(v, 4) for k, v in all_scores.items()},
                is_trouble=(top_label in TROUBLE_LABELS and top_score >= threshold),
            ),
        )

    for idx in empty_indices:
        c = comments[idx]
        logger.debug("Empty text for comment %s; assigning neutral label", c.get("id"))
        results[idx] = ClassifiedComment(
            comment_id=c.get("id", ""),
            author=c.get("author", ""),
            text=c.get("text", ""),
            timestamps_in_text=c.get("timestamps_seconds", []),
            classification=ClassificationResult(
                predicted_label=_NEUTRAL_LABEL,
                confidence=1.0,
                all_scores=_NEUTRAL_SCORES.copy(),
                is_trouble=False,
            ),
        )

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def classify_video_comments(
    url: str,
    threshold: float | None = None,
    force: bool = False,
) -> tuple[list[ClassifiedComment], bool]:
    """Classify timestamped comments for a video that has been ingested.

    Returns (classified_comments, from_cache).
    Raises VideoNotReadyError, CommentsNotReadyError, or ModelLoadError.
    """
    _threshold = threshold if threshold is not None else settings.CLASSIFIER_CONFIDENCE_THRESHOLD

    video = state.get_video(url)
    if video is None:
        raise VideoNotReadyError(f"No ingested video for URL: {url}")

    raw_comments: list[dict] = video.get("comments", [])
    if not raw_comments:
        raise CommentsNotReadyError(
            "No comments found. Ensure YOUTUBE_API_KEY is set and re-ingest the video."
        )

    if video.get("classified_comments") and not force:
        cached = [ClassifiedComment.model_validate(c) for c in video["classified_comments"]]
        return cached, True

    if not raw_comments:
        logger.warning("No comments found for %s; storing empty result", url)
        video["classified_comments"] = []
        return [], False

    timestamped_count = sum(1 for c in raw_comments if c.get("has_timestamp"))
    logger.info(
        "Classification started: url=%s total=%d timestamped=%d threshold=%.2f device=%s",
        url, len(raw_comments), timestamped_count, _threshold, _classifier_device or "unloaded",
    )
    t0 = time.monotonic()

    classified = classify_comments(raw_comments, threshold=_threshold)

    video["classified_comments"] = [c.model_dump() for c in classified]

    elapsed = time.monotonic() - t0
    trouble = sum(1 for c in classified if c.classification.is_trouble)
    logger.info(
        "Classification complete: %d classified, %d trouble, %.1fs",
        len(classified), trouble, elapsed,
    )

    return classified, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_label_distribution(classified: list[ClassifiedComment]) -> dict[str, int]:
    """Return a count of comments per predicted_label."""
    dist: dict[str, int] = {}
    for c in classified:
        lbl = c.classification.predicted_label
        dist[lbl] = dist.get(lbl, 0) + 1
    return dist
