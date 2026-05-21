"""Full pipeline orchestrator. Runs all stages sequentially as a background task.

Each stage writes its output into app.state so that individual endpoints remain
usable for incremental runs. The orchestrator is purely additive — it never
overwrites results that already exist unless force=True.
"""

import logging
import traceback

from app import state
from app.config import settings
from app.models import (
    AnalysisResult,
    VideoMetadata,
    ConceptReport,
    ConceptSynthesis,
    MatchedComment,
    SentimentSummary,
    CommentSample,
    MatchResult,
    DISCLAIMERS,
)
from app.pipeline.ingest import ingest_video, IngestError
from app.pipeline.transcribe import transcribe_video, TranscribeError
from app.pipeline.concepts import extract_concepts, ConceptExtractionFailedError
from app.pipeline.classify import classify_video_comments, ClassificationError, CommentsNotReadyError
from app.pipeline.match import match_video, MatchingError, ClassifiedCommentsNotReadyError, ConceptsNotReadyError
from app.pipeline.sentiment import analyze_video_sentiment, SentimentError
from app.pipeline.synthesize import synthesize_video, SynthesisError
from app.pipeline.llm import OllamaUnreachableError

logger = logging.getLogger(__name__)

# Stage names and their completion progress (0–100)
STAGES = [
    ("ingest", 10),
    ("transcribe", 25),
    ("extract_concepts", 45),
    ("classify", 60),
    ("match", 75),
    ("sentiment", 85),
    ("synthesize", 100),
]

STAGE_PROGRESS = {name: pct for name, pct in STAGES}
PREV_PROGRESS = {name: STAGES[i - 1][1] if i > 0 else 0 for i, (name, _) in enumerate(STAGES)}


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def run_pipeline(job_id: str, url: str, force: bool = False) -> None:
    """Execute the full pipeline for *url*, updating job state along the way.

    Never raises — all exceptions are caught and recorded in state.
    """
    def _set(stage: str, progress: int | None = None, **kw):
        fields = {"stage": stage}
        if progress is not None:
            fields["progress"] = progress
        fields.update(kw)
        state.update_job(job_id, **fields)

    _set("ingest", progress=0, status="running")

    try:
        # ── 1. Ingest ───────────────────────────────────────────────────────
        _set("ingest", progress=PREV_PROGRESS["ingest"])
        try:
            ingest_video(url)
        except IngestError as exc:
            _set("ingest", status="failed", error_message=str(exc))
            return
        _set("ingest", progress=STAGE_PROGRESS["ingest"])

        # ── 2. Transcribe ───────────────────────────────────────────────────
        _set("transcribe", progress=PREV_PROGRESS["transcribe"])
        try:
            transcribe_video(url=url, model_size=settings.WHISPER_MODEL_SIZE, force=force)
        except TranscribeError as exc:
            _set("transcribe", status="failed", error_message=str(exc))
            return
        _set("transcribe", progress=STAGE_PROGRESS["transcribe"])

        # ── 3. Extract concepts ─────────────────────────────────────────────
        _set("extract_concepts", progress=PREV_PROGRESS["extract_concepts"])
        try:
            extract_concepts(url=url, model=settings.OLLAMA_MODEL, force=force)
        except OllamaUnreachableError as exc:
            _set("extract_concepts", status="failed", error_message=str(exc))
            return
        except ConceptExtractionFailedError as exc:
            _set("extract_concepts", status="failed", error_message=str(exc))
            return
        _set("extract_concepts", progress=STAGE_PROGRESS["extract_concepts"])

        # ── 4. Classify comments ────────────────────────────────────────────
        _set("classify", progress=PREV_PROGRESS["classify"])
        try:
            classify_video_comments(
                url=url,
                threshold=settings.CLASSIFIER_CONFIDENCE_THRESHOLD,
                force=force,
            )
        except CommentsNotReadyError as exc:
            logger.warning("Classify skipped (no comments): %s", exc)
        except ClassificationError as exc:
            _set("classify", status="failed", error_message=str(exc))
            return
        _set("classify", progress=STAGE_PROGRESS["classify"])

        # ── 5. Match ────────────────────────────────────────────────────────
        _set("match", progress=PREV_PROGRESS["match"])
        try:
            match_video(url=url, threshold=settings.SIMILARITY_THRESHOLD, force=force)
        except (ClassifiedCommentsNotReadyError, ConceptsNotReadyError) as exc:
            logger.warning("Match skipped: %s", exc)
        except MatchingError as exc:
            _set("match", status="failed", error_message=str(exc))
            return
        _set("match", progress=STAGE_PROGRESS["match"])

        # ── 6. Sentiment ────────────────────────────────────────────────────
        _set("sentiment", progress=PREV_PROGRESS["sentiment"])
        try:
            analyze_video_sentiment(url=url, force=force)
        except SentimentError as exc:
            logger.warning("Sentiment skipped: %s", exc)
        _set("sentiment", progress=STAGE_PROGRESS["sentiment"])

        # ── 7. Synthesize ───────────────────────────────────────────────────
        _set("synthesize", progress=PREV_PROGRESS["synthesize"])
        try:
            synthesize_video(url=url, model=settings.OLLAMA_MODEL, force=force)
        except OllamaUnreachableError as exc:
            _set("synthesize", status="failed", error_message=str(exc))
            return
        except SynthesisError as exc:
            logger.warning("Synthesis skipped: %s", exc)
        _set("synthesize", progress=STAGE_PROGRESS["synthesize"])

        # ── Done ────────────────────────────────────────────────────────────
        result = assemble_result(url)

        # sentiment_labeled is only needed by assemble_result — drop it now
        # so it doesn't sit in memory for the lifetime of the video entry.
        video = state.get_video(url)
        if video:
            video.pop("sentiment_labeled", None)

        state.update_job(
            job_id,
            status="completed",
            stage="done",
            progress=100,
            result=result.model_dump(mode="json"),
        )
        logger.info("Pipeline complete for job %s url=%s", job_id, url)

    except Exception as exc:
        logger.error(
            "Unexpected pipeline failure for job %s: %s\n%s",
            job_id, exc, traceback.format_exc(),
        )
        state.update_job(job_id, status="failed", error_message=str(exc))


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def assemble_result(url: str) -> AnalysisResult:
    """Build an AnalysisResult from current state for *url*."""
    video = state.get_video(url)
    if video is None:
        raise ValueError(f"No video in state for URL: {url}")

    metadata = VideoMetadata(
        video_id=video.get("id", ""),
        title=video.get("title", ""),
        uploader=video.get("uploader", ""),
        duration=video.get("duration") or 0.0,
        view_count=video.get("view_count"),
        upload_date=video.get("upload_date"),
    )

    # Concepts
    concepts_raw = video.get("concepts") or []

    # Matches — used to build ConceptReport
    matches_raw = video.get("matches")
    match_result: MatchResult | None = None
    if matches_raw is not None:
        try:
            match_result = MatchResult.model_validate(matches_raw)
        except Exception:
            pass

    # Syntheses — keyed by concept_name
    syntheses_raw = video.get("syntheses") or []
    synthesis_map: dict[str, ConceptSynthesis] = {}
    for s in syntheses_raw:
        try:
            cs = ConceptSynthesis.model_validate(s)
            synthesis_map[cs.concept_name] = cs
        except Exception:
            pass

    # Build a lookup: concept_name → ConceptMatch for O(1) access
    concept_match_map: dict[str, list[MatchedComment]] = {}
    if match_result:
        for cm in match_result.concept_matches:
            concept_match_map[cm.concept.concept] = cm.matched_comments

    concept_reports: list[ConceptReport] = []
    for c_raw in concepts_raw:
        concept_name = c_raw.get("concept", "")
        matched_comments = concept_match_map.get(concept_name, [])
        synthesis = synthesis_map.get(concept_name)

        concept_reports.append(ConceptReport(
            concept_name=concept_name,
            keywords=c_raw.get("keywords", []),
            explanation=c_raw.get("explanation", ""),
            timestamp_start=c_raw.get("timestamp_start", 0.0),
            timestamp_end=c_raw.get("timestamp_end", 0.0),
            matched_comment_count=len(matched_comments),
            matched_comments=matched_comments,
            synthesis=synthesis,
        ))

    # Unmatched trouble comments
    unmatched_comments: list[MatchedComment] = []
    if match_result:
        unmatched_comments = list(match_result.unmatched.comments)

    # Sentiment
    sentiment: SentimentSummary | None = None
    sentiment_raw = video.get("sentiment")
    if sentiment_raw:
        try:
            sentiment = SentimentSummary.model_validate(sentiment_raw)
        except Exception:
            pass

    # Comment counts
    all_comments: list[dict] = video.get("comments") or []
    total_comments = len(all_comments)
    timestamped_comments = sum(1 for c in all_comments if c.get("has_timestamp"))

    _MAX_SAMPLES = 15

    trouble_count = 0
    label_distribution: dict[str, int] = {}
    label_comment_samples: dict[str, list[CommentSample]] = {}
    classified_raw = video.get("classified_comments") or []
    for c in classified_raw:
        classification = c.get("classification") or {}
        if classification.get("is_trouble"):
            trouble_count += 1
        label = classification.get("predicted_label", "")
        if label:
            label_distribution[label] = label_distribution.get(label, 0) + 1
            bucket = label_comment_samples.setdefault(label, [])
            if len(bucket) < _MAX_SAMPLES:
                bucket.append(CommentSample(
                    text=c.get("text", ""),
                    author=c.get("author", ""),
                ))

    sentiment_comment_samples: dict[str, list[CommentSample]] = {}
    for item in (video.get("sentiment_labeled") or []):
        label = item.get("label", "")
        if not label:
            continue
        bucket = sentiment_comment_samples.setdefault(label, [])
        if len(bucket) < _MAX_SAMPLES:
            bucket.append(CommentSample(
                text=item.get("text", ""),
                author=item.get("author", ""),
            ))

    return AnalysisResult(
        url=url,
        metadata=metadata,
        total_comments=total_comments,
        timestamped_comments=timestamped_comments,
        trouble_comment_count=trouble_count,
        unmatched_trouble_count=len(unmatched_comments),
        unmatched_trouble_comments=unmatched_comments,
        concept_reports=concept_reports,
        sentiment=sentiment,
        label_distribution=label_distribution,
        label_comment_samples=label_comment_samples,
        sentiment_comment_samples=sentiment_comment_samples,
        disclaimers=list(DISCLAIMERS),
    )
