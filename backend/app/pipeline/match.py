"""Match trouble comments to concepts via cosine similarity of sentence embeddings."""

import logging
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app import state
from app.config import settings
from app.models import (
    Concept,
    ClassifiedComment,
    MatchedComment,
    ConceptMatch,
    UnmatchedBucket,
    MatchResult,
)
from app.pipeline.embed import embed_texts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MatchingError(Exception):
    pass

class VideoNotReadyError(MatchingError):
    pass

class ConceptsNotReadyError(MatchingError):
    pass

class ClassifiedCommentsNotReadyError(MatchingError):
    pass

class EmbeddingFailedError(MatchingError):
    pass


# Higher threshold for comments without a timestamp anchor — similarity alone
# must be strong enough to justify the match without positional evidence.
UNTIMESTAMPED_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def concept_to_text(concept: Concept) -> str:
    """Serialize a concept for embedding as '{name}. {explanation}'.

    Including the explanation gives the model more semantic context than
    the short name alone, which helps when comments use different surface
    words than the concept label.
    """
    return f"{concept.concept}. {concept.explanation}"


def comment_to_text(comment: ClassifiedComment) -> str:
    """Return the raw comment text for embedding."""
    return comment.text


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_comments_to_concepts(
    trouble_comments: list[ClassifiedComment],
    concepts: list[Concept],
    threshold: float,
    has_timestamp: bool = True,
) -> MatchResult:
    """Match each trouble comment to its best-matching concept.

    Returns a MatchResult with per-concept buckets and an unmatched bucket.
    Matched comments within each bucket are sorted by similarity descending.
    Concept matches are sorted by timestamp_start ascending (video order).
    Unmatched comments are sorted by similarity_score descending.
    """
    # --- empty trouble_comments ---
    if not trouble_comments:
        concept_matches = sorted(
            [ConceptMatch(concept=c, matched_comments=[]) for c in concepts],
            key=lambda cm: cm.concept.timestamp_start,
        )
        return MatchResult(
            concept_matches=concept_matches,
            unmatched=UnmatchedBucket(comments=[]),
        )

    # --- empty concepts: everything goes to unmatched ---
    if not concepts:
        unmatched = [_to_matched(tc, 0.0, has_timestamp) for tc in trouble_comments]
        return MatchResult(
            concept_matches=[],
            unmatched=UnmatchedBucket(comments=unmatched),
        )

    # --- embed both sides ---
    concept_texts = [concept_to_text(c) for c in concepts]
    comment_texts = [comment_to_text(c) for c in trouble_comments]

    concept_embs = embed_texts(concept_texts)   # (n_concepts, 768)
    comment_embs = embed_texts(comment_texts)   # (n_comments, 768)

    # sim[i, j] = similarity between comment i and concept j
    sim = cosine_similarity(comment_embs, concept_embs)  # (n_comments, n_concepts)

    best_concept_idx = np.argmax(sim, axis=1)                                        # (n_comments,)
    best_scores = sim[np.arange(len(trouble_comments)), best_concept_idx].tolist()   # (n_comments,)

    # log range info for threshold tuning
    logger.debug(
        "Similarity scores — min=%.3f max=%.3f mean=%.3f",
        float(np.min(best_scores)), float(np.max(best_scores)), float(np.mean(best_scores)),
    )

    # --- bucket by concept ---
    concept_buckets: dict[int, list[MatchedComment]] = {i: [] for i in range(len(concepts))}
    unmatched_list: list[MatchedComment] = []

    for i, (ci, score) in enumerate(zip(best_concept_idx.tolist(), best_scores)):
        mc = _to_matched(trouble_comments[i], float(score), has_timestamp)
        if score >= threshold:
            concept_buckets[int(ci)].append(mc)
        else:
            unmatched_list.append(mc)

    # sort within each bucket
    concept_matches = []
    for i, concept in enumerate(concepts):
        bucket = sorted(concept_buckets[i], key=lambda m: m.similarity_score, reverse=True)
        concept_matches.append(ConceptMatch(concept=concept, matched_comments=bucket))

    concept_matches.sort(key=lambda cm: cm.concept.timestamp_start)
    unmatched_list.sort(key=lambda m: m.similarity_score, reverse=True)

    return MatchResult(
        concept_matches=concept_matches,
        unmatched=UnmatchedBucket(comments=unmatched_list),
    )


def _to_matched(comment: ClassifiedComment, score: float, has_timestamp: bool = True) -> MatchedComment:
    return MatchedComment(
        comment_id=comment.comment_id,
        author=comment.author,
        text=comment.text,
        timestamps_in_text=comment.timestamps_in_text,
        predicted_label=comment.classification.predicted_label,
        classification_confidence=comment.classification.confidence,
        similarity_score=round(score, 4),
        has_timestamp=has_timestamp,
    )


def _merge_results(r1: MatchResult, r2: MatchResult) -> MatchResult:
    """Merge two MatchResults that cover the same concept list."""
    concept_matches = []
    for cm1, cm2 in zip(r1.concept_matches, r2.concept_matches):
        combined = sorted(
            cm1.matched_comments + cm2.matched_comments,
            key=lambda m: m.similarity_score,
            reverse=True,
        )
        concept_matches.append(ConceptMatch(concept=cm1.concept, matched_comments=combined))

    unmatched = sorted(
        r1.unmatched.comments + r2.unmatched.comments,
        key=lambda m: m.similarity_score,
        reverse=True,
    )
    return MatchResult(concept_matches=concept_matches, unmatched=UnmatchedBucket(comments=unmatched))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def match_video(
    url: str,
    threshold: float | None = None,
    force: bool = False,
) -> tuple[MatchResult, bool]:
    """Match trouble comments to concepts for an already-processed video.

    Returns (MatchResult, from_cache).
    Raises VideoNotReadyError, ConceptsNotReadyError,
    ClassifiedCommentsNotReadyError, or EmbeddingFailedError.
    """
    _threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    video = state.get_video(url)
    if video is None:
        raise VideoNotReadyError(f"No ingested video for URL: {url}")

    concepts_data = video.get("concepts")
    if concepts_data is None:
        raise ConceptsNotReadyError(
            "Concepts have not been extracted. Call POST /extract-concepts first."
        )

    cc_data = video.get("classified_comments")
    if cc_data is None:
        raise ClassifiedCommentsNotReadyError(
            "Comments have not been classified. Call POST /classify first."
        )

    if video.get("matches") and not force:
        return MatchResult.model_validate(video["matches"]), True

    concepts = [Concept.model_validate(c) for c in concepts_data]
    all_classified = [ClassifiedComment.model_validate(c) for c in cc_data]

    ts_trouble    = [c for c in all_classified if c.classification.is_trouble and c.timestamps_in_text]
    nots_trouble  = [c for c in all_classified if c.classification.is_trouble and not c.timestamps_in_text]

    logger.info(
        "Matching started: url=%s concepts=%d trouble_ts=%d trouble_nots=%d threshold=%.2f/%.2f",
        url, len(concepts), len(ts_trouble), len(nots_trouble), _threshold, UNTIMESTAMPED_THRESHOLD,
    )
    t0 = time.monotonic()

    try:
        result_ts   = match_comments_to_concepts(ts_trouble,   concepts, _threshold,            has_timestamp=True)
        result_nots = match_comments_to_concepts(nots_trouble,  concepts, UNTIMESTAMPED_THRESHOLD, has_timestamp=False)
        result = _merge_results(result_ts, result_nots)
    except Exception as exc:
        raise EmbeddingFailedError(str(exc)) from exc

    video["matches"] = result.model_dump()

    elapsed = time.monotonic() - t0
    matched = sum(len(cm.matched_comments) for cm in result.concept_matches)
    unmatched = len(result.unmatched.comments)
    concepts_hit = sum(1 for cm in result.concept_matches if cm.matched_comments)

    logger.info(
        "Matching complete: matched=%d (ts=%d nots=%d) unmatched=%d concepts_hit=%d in %.1fs",
        matched,
        sum(len(cm.matched_comments) for cm in result_ts.concept_matches),
        sum(len(cm.matched_comments) for cm in result_nots.concept_matches),
        unmatched, concepts_hit, elapsed,
    )

    return result, False
