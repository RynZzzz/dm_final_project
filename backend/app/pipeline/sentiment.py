"""Aggregate sentiment analysis across all comments. Pretrained Twitter RoBERTa
model. Used as a context layer, not for fine-grained analysis."""

import logging
import time

from app import state
from app.models import SentimentSummary

logger = logging.getLogger(__name__)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

DISCLAIMER = (
    "Sentiment reflects the tone of comments that exist on the video, "
    "not all viewers. YouTube comment sections tend to skew positive: "
    "viewers who disliked the video are more likely to leave silently "
    "than to comment. Treat this as ambient context, not a diagnostic."
)

# Defensive label normalization — transformers version determines naming
LABEL_NORMALIZE = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

_sentiment_pipeline = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SentimentError(Exception):
    pass

class VideoNotReadyError(SentimentError):
    pass

class CommentsNotReadyError(SentimentError):
    pass


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_sentiment_pipeline():
    """Return a cached HuggingFace sentiment pipeline.

    Auto-detects CUDA with CPU fallback. Loads on first call.
    """
    global _sentiment_pipeline

    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    import torch
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    device_name = "cuda" if device == 0 else "cpu"
    dtype = torch.float16 if device == 0 else torch.float32

    logger.info("Loading sentiment model '%s' on %s (%s)", MODEL_NAME, device_name, dtype)
    _sentiment_pipeline = hf_pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
        torch_dtype=dtype,
        truncation=True,
        max_length=512,
    )
    logger.info("Sentiment model ready on %s", device_name)
    return _sentiment_pipeline


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_sentiment(
    comments: list[dict],
    batch_size: int = 64,
) -> tuple[SentimentSummary, list[dict]]:
    """Run sentiment analysis on all comment dicts.

    Returns (SentimentSummary, labeled_comments) where labeled_comments is a
    list of {text, author, label} dicts for every processed comment.
    Empty-text comments are skipped. Returns zeroed summary if nothing valid.
    """
    pipe = get_sentiment_pipeline()
    valid = [c for c in comments if c.get("text", "").strip()]

    if not valid:
        return SentimentSummary(
            positive_pct=0.0,
            neutral_pct=0.0,
            negative_pct=0.0,
            sample_size=0,
            disclaimer=DISCLAIMER,
        ), []

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    labeled: list[dict] = []

    for start in range(0, len(valid), batch_size):
        batch_comments = valid[start : start + batch_size]
        batch_texts = [c["text"] for c in batch_comments]
        outputs = pipe(batch_texts)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for c, o in zip(batch_comments, outputs):
            label = LABEL_NORMALIZE.get(o["label"], "neutral")
            counts[label] += 1
            labeled.append({"text": c["text"], "author": c.get("author", ""), "label": label})

        logger.debug(
            "Sentiment batch %d–%d / %d processed",
            start + 1, start + len(batch_texts), len(valid),
        )

    total = sum(counts.values())
    summary = SentimentSummary(
        positive_pct=round(counts["positive"] / total * 100, 1),
        neutral_pct=round(counts["neutral"] / total * 100, 1),
        negative_pct=round(counts["negative"] / total * 100, 1),
        sample_size=total,
        disclaimer=DISCLAIMER,
    )
    return summary, labeled


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_video_sentiment(
    url: str,
    force: bool = False,
) -> tuple[SentimentSummary, bool]:
    """Run sentiment analysis for a video that has been ingested.

    Returns (SentimentSummary, from_cache).
    Raises VideoNotReadyError or CommentsNotReadyError.
    """
    video = state.get_video(url)
    if video is None:
        raise VideoNotReadyError(f"No ingested video for URL: {url}")

    comments: list[dict] = video.get("comments", [])
    if not comments:
        raise CommentsNotReadyError(
            "No comments found. Ensure YOUTUBE_API_KEY is set and re-ingest."
        )

    if video.get("sentiment") and not force:
        return SentimentSummary.model_validate(video["sentiment"]), True

    t0 = time.monotonic()
    summary, labeled = analyze_sentiment(comments)
    elapsed = time.monotonic() - t0

    video["sentiment"] = summary.model_dump()
    video["sentiment_labeled"] = labeled

    logger.info(
        "Sentiment complete: sample=%d positive=%.1f%% neutral=%.1f%% negative=%.1f%% in %.1fs",
        summary.sample_size,
        summary.positive_pct,
        summary.neutral_pct,
        summary.negative_pct,
        elapsed,
    )

    return summary, False
