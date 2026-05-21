"""Per-concept synthesis via Llama 3. Strictly grounded: summarizes matched
comments and quotes them. Forbids recommendations or invented details."""

import json
import logging
import re
import time

from app import state
from app.config import settings
from app.models import (
    Concept,
    MatchedComment,
    MatchResult,
    Transcript,
    ConceptSynthesis,
)
from app.pipeline import llm
from app.pipeline.llm import OllamaUnreachableError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SynthesisError(Exception):
    pass

class VideoNotReadyError(SynthesisError):
    pass

class TranscriptNotReadyError(SynthesisError):
    pass

class MatchesNotReadyError(SynthesisError):
    pass


# ---------------------------------------------------------------------------
# Transcript excerpt
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def get_transcript_excerpt(
    transcript: Transcript,
    start: float,
    end: float,
    pad_seconds: float = 5.0,
) -> str:
    """Concatenate transcript segments overlapping [start-pad, end+pad]."""
    t0 = start - pad_seconds
    t1 = end + pad_seconds
    segs = [s for s in transcript.segments if s.end >= t0 and s.start <= t1]
    return " ".join(s.text.strip() for s in segs)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    concept: Concept,
    matched_comments: list[MatchedComment],
    transcript_excerpt: str,
) -> str:
    """Build the Llama 3 prompt for one concept's synthesis."""
    kw = ", ".join(concept.keywords) if concept.keywords else "none"

    comment_lines = []
    for c in matched_comments[:10]:  # cap at 10 to stay within context window
        ts = f"[{_fmt_ts(c.timestamps_in_text[0])}] " if c.timestamps_in_text else ""
        comment_lines.append(f"- {ts}{c.text}")
    comments_block = "\n".join(comment_lines)

    return f"""You are summarizing viewer feedback on a specific concept from an educational video.

CONCEPT: {concept.concept}
KEYWORDS: {kw}
EXPLANATION: {concept.explanation}

TRANSCRIPT EXCERPT (covering this concept's time range):
{transcript_excerpt}

VIEWER COMMENTS ABOUT THIS CONCEPT:
{comments_block}

Write a 2-3 sentence summary of what these viewers expressed about this concept.
Tone: neutral, descriptive, factual. No recommendations. No invented claims.

RULES (strictly enforced):
- Summarize only what the comments actually express. Do not add interpretation.
- Do NOT write recommendations ("the creator should...", "this needs more...")
- Do NOT write generic pedagogical opinions ("this should be explained more slowly")
- Do NOT quote comments verbatim inside the summary itself (quotes go in quoted_evidence)
- Do NOT invent claims with no source in the comments

GOOD summary: "Several commenters expressed difficulty with zero-based indexing, \
particularly why the first element is at position 0. A few asked clarifying questions \
comparing this convention to natural counting."

BAD summary (recommendation): "The creator should slow down and add a visual diagram."
BAD summary (invented): "Most viewers under 25 found the indexing confusing." (no source)

Return ONLY valid JSON with exactly two fields:
  "summary"         — 2-3 sentence neutral factual description of commenter reactions
  "quoted_evidence" — list of 1-3 short verbatim quotes from the comments above

Example output:
{{"summary": "Several viewers struggled with...", "quoted_evidence": ["why is it 0?", "so confusing"]}}

JSON:"""


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _parse_synthesis_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No JSON object found in synthesis response")


def _normalize_ws(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Per-concept synthesis
# ---------------------------------------------------------------------------

def synthesize_concept(
    concept: Concept,
    matched_comments: list[MatchedComment],
    transcript: Transcript,
    model: str,
) -> ConceptSynthesis:
    """Generate a synthesis summary for one concept.

    Retries once with a stricter JSON preamble if parse fails.
    Falls back to a placeholder if synthesis cannot be produced.
    """
    excerpt = get_transcript_excerpt(transcript, concept.timestamp_start, concept.timestamp_end)
    prompt = build_synthesis_prompt(concept, matched_comments, excerpt)

    comment_texts = [c.text for c in matched_comments]

    data: dict | None = None
    for attempt in range(2):
        actual_prompt = ("Return ONLY valid JSON.\n\n" + prompt) if attempt else prompt
        try:
            raw = llm.generate(actual_prompt, model=model, format_json=True, temperature=0.3)
            data = _parse_synthesis_json(raw)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == 0:
                logger.debug(
                    "Synthesis parse failed for '%s' (attempt 1), retrying: %s",
                    concept.concept, exc,
                )
            else:
                logger.warning(
                    "Synthesis parse failed for '%s' after retry", concept.concept
                )

    # Fallback when parsing failed completely
    if data is None:
        fallback_quote = matched_comments[0].text[:200] if matched_comments else ""
        return ConceptSynthesis(
            concept_name=concept.concept,
            summary="(Summary generation failed for this concept.)",
            quoted_evidence=[fallback_quote] if fallback_quote else [],
            matched_comment_count=len(matched_comments),
        )

    # Validate summary
    summary = str(data.get("summary", "")).strip()
    if not summary:
        summary = "(Summary generation failed for this concept.)"
        logger.warning("Empty summary returned for concept '%s'", concept.concept)

    # Validate and ground quoted evidence
    raw_quotes = data.get("quoted_evidence", [])
    if not isinstance(raw_quotes, list):
        raw_quotes = []

    grounded: list[str] = []
    for q in raw_quotes[:3]:
        q = str(q).strip()
        if not q:
            continue
        q_norm = _normalize_ws(q)
        # Check that the quote appears in at least one comment
        if any(q_norm in _normalize_ws(ct) for ct in comment_texts):
            grounded.append(q)
        else:
            logger.debug(
                "Dropping hallucinated quote for '%s': '%.60s…'", concept.concept, q
            )

    # If all quotes were dropped, use the first comment as fallback evidence
    if not grounded and matched_comments:
        grounded = [matched_comments[0].text[:200]]

    return ConceptSynthesis(
        concept_name=concept.concept,
        summary=summary,
        quoted_evidence=grounded,
        matched_comment_count=len(matched_comments),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def synthesize_video(
    url: str,
    model: str | None = None,
    force: bool = False,
    min_matches: int = 2,
) -> tuple[list[ConceptSynthesis], int, bool]:
    """Generate syntheses for all concepts with >= min_matches matched comments.

    Returns (syntheses, skipped_count, from_cache).
    Raises VideoNotReadyError, TranscriptNotReadyError, MatchesNotReadyError.
    OllamaUnreachableError propagates directly so the endpoint maps it to 503.
    """
    video = state.get_video(url)
    if video is None:
        raise VideoNotReadyError(f"No ingested video for URL: {url}")

    if not video.get("transcript"):
        raise TranscriptNotReadyError(
            "No transcript found. Call POST /transcribe first."
        )

    if video.get("matches") is None:
        raise MatchesNotReadyError(
            "No match results found. Call POST /match first."
        )

    if video.get("syntheses") is not None and not force:
        cached = [ConceptSynthesis.model_validate(s) for s in video["syntheses"]]
        skipped = video.get("_syntheses_skipped", 0)
        return cached, skipped, True

    _model = model or settings.OLLAMA_MODEL
    transcript = Transcript.model_validate(video["transcript"])
    match_result = MatchResult.model_validate(video["matches"])

    eligible = [cm for cm in match_result.concept_matches if len(cm.matched_comments) >= min_matches]
    skipped = len(match_result.concept_matches) - len(eligible)

    logger.info(
        "Synthesis started: url=%s eligible=%d skipped=%d model=%s",
        url, len(eligible), skipped, _model,
    )
    t0 = time.monotonic()

    syntheses: list[ConceptSynthesis] = []
    fallback_count = 0
    first_call_done = False

    for cm in eligible:
        try:
            syn = synthesize_concept(cm.concept, cm.matched_comments, transcript, _model)
            if syn.summary.startswith("(Summary generation failed"):
                fallback_count += 1
            syntheses.append(syn)
            first_call_done = True
        except OllamaUnreachableError:
            if not first_call_done:
                raise  # fail fast — Ollama is down before we produced anything
            logger.warning(
                "Ollama became unreachable mid-synthesis after %d concepts; "
                "returning partial results",
                len(syntheses),
            )
            break
        except Exception as exc:
            logger.warning(
                "Synthesis failed for concept '%s': %s", cm.concept.concept, exc
            )
            fallback_count += 1
            syntheses.append(ConceptSynthesis(
                concept_name=cm.concept.concept,
                summary="(Summary generation failed for this concept.)",
                quoted_evidence=[cm.matched_comments[0].text[:200]] if cm.matched_comments else [],
                matched_comment_count=len(cm.matched_comments),
            ))

    video["syntheses"] = [s.model_dump() for s in syntheses]
    video["_syntheses_skipped"] = skipped

    elapsed = time.monotonic() - t0
    logger.info(
        "Synthesis complete: %d synthesized (%d fallbacks), %d skipped in %.1fs",
        len(syntheses), fallback_count, skipped, elapsed,
    )

    return syntheses, skipped, False
