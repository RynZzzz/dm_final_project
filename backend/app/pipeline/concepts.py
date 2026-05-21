"""Concept extraction using Ollama (Llama 3).

Chunks the transcript, sends each chunk to the LLM with a structured
prompt, validates and grounds each returned concept against the source
text, deduplicates across chunks, and stores results in state.
"""

import json
import logging
import re
import string
import time

from app import state
from app.config import settings
from app.models import Concept, Transcript
from app.pipeline import llm
from app.pipeline.llm import OllamaError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConceptExtractionError(Exception):
    pass

class VideoNotReadyError(ConceptExtractionError):
    pass

class TranscriptNotReadyError(ConceptExtractionError):
    pass

class ConceptExtractionFailedError(ConceptExtractionError):
    pass

class ConceptParseError(ConceptExtractionError):
    pass


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(
    transcript: Transcript,
    max_words: int = 400,
    overlap_fraction: float = 0.10,
) -> list[dict]:
    """Split a transcript into word-bounded chunks with overlap.

    Returns a list of chunk dicts:
      index, text, start, end, segment_indices (inclusive tuple)

    If the whole transcript is under max_words, returns a single chunk.
    Overlap allows the LLM to see concepts that straddle chunk boundaries.
    """
    segments = transcript.segments
    if not segments:
        return []

    total_words = sum(len(s.text.split()) for s in segments)
    if total_words <= max_words:
        return [{
            "index": 0,
            "text": " ".join(s.text.strip() for s in segments),
            "start": segments[0].start,
            "end": segments[-1].end,
            "segment_indices": (0, len(segments) - 1),
        }]

    overlap_words = int(max_words * overlap_fraction)
    chunks: list[dict] = []
    chunk_start = 0

    while chunk_start < len(segments):
        word_count = 0
        chunk_end = chunk_start

        while chunk_end < len(segments) and word_count < max_words:
            word_count += len(segments[chunk_end].text.split())
            chunk_end += 1

        chunk_segs = segments[chunk_start:chunk_end]
        chunks.append({
            "index": len(chunks),
            "text": " ".join(s.text.strip() for s in chunk_segs),
            "start": chunk_segs[0].start,
            "end": chunk_segs[-1].end,
            "segment_indices": (chunk_start, chunk_end - 1),
        })

        if chunk_end >= len(segments):
            break

        # Rewind from chunk_end to find the overlap start point
        overlap_count = 0
        new_start = chunk_end
        while new_start > chunk_start + 1 and overlap_count < overlap_words:
            new_start -= 1
            overlap_count += len(segments[new_start].text.split())

        chunk_start = max(chunk_start + 1, new_start)  # always advance

    return chunks


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_concept_prompt(chunk: dict, transcript_segments: list) -> str:
    """Build the full prompt for one chunk, with inline segment timestamps."""
    start_idx, end_idx = chunk["segment_indices"]
    segs = transcript_segments[start_idx : end_idx + 1]

    annotated = "\n".join(
        f"[{_fmt_ts(seg.start)}] {seg.text.strip()}" for seg in segs
    )

    t0, t1 = chunk["start"], chunk["end"]

    return f"""You are a pedagogical analyst extracting the concepts taught in an educational video.

A CONCEPT is a teachable unit at roughly textbook-section level. Preserve compound terms as single units.

Good concept names: "for loop", "nested loop", "zero-based indexing", "recursive base case", "gradient descent", "differential calculus"
Too broad: "programming", "math", "algorithms"
Too narrow: "using the range function with step 2"

This chunk covers {_fmt_ts(t0)} to {_fmt_ts(t1)} (seconds {t0:.1f}–{t1:.1f}).

TRANSCRIPT:
{annotated}

Extract every distinct concept TAUGHT in this chunk. For each concept return:
  "concept"         — 2–6 word name (keep compound terms together)
  "keywords"        — list of related terms
  "explanation"     — exactly 1 sentence explaining what is taught
  "timestamp_start" — seconds, must be within {t0:.1f}–{t1:.1f}
  "timestamp_end"   — seconds, must be within {t0:.1f}–{t1:.1f}

Rules:
- Do not invent concepts absent from this chunk.
- Return an empty list [] if the chunk has no teaching content (intro, outro, ads).

Return ONLY valid JSON in this exact format:
{{"concepts": [
  {{
    "concept": "for loop",
    "keywords": ["iteration", "loop variable", "range"],
    "explanation": "A for loop repeats a block of code for each item in a sequence.",
    "timestamp_start": {t0:.1f},
    "timestamp_end": {t1:.1f}
  }}
]}}

JSON:"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_concept_response(raw: str) -> list[dict]:
    """Defensively parse the LLM's JSON output into a list of dicts.

    Strips markdown fences, tries direct parse, then regex extraction.
    Unwraps {"concepts": [...]} if the model wrapped the array.
    """
    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip()).strip()

    # Direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back: find the first [...] or {...} block
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if not match:
            raise ConceptParseError("No JSON structure found in LLM response")
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ConceptParseError(f"JSON parse failed: {exc}") from exc

    # Unwrap any single-key object whose value is a list
    # e.g. {"concepts": [...]}, {"items": [...]}, {"result": [...]}
    if isinstance(data, dict):
        list_values = [v for v in data.values() if isinstance(v, list)]
        if list_values:
            data = list_values[0]
        else:
            raise ConceptParseError(f"JSON object has no list value: {list(data.keys())}")

    if not isinstance(data, list):
        raise ConceptParseError(f"Expected a JSON array, got {type(data).__name__}")

    return data


# ---------------------------------------------------------------------------
# Per-chunk extraction
# ---------------------------------------------------------------------------

def _normalize_ws(text: str) -> str:
    return " ".join(text.lower().split())


def _is_grounded(concept_name: str, evidence: str, chunk_norm: str) -> bool:
    """Return True if the concept appears to be grounded in the chunk text.

    Two-level check:
    1. Strict  — evidence string appears verbatim (whitespace-normalized) in chunk
    2. Fallback — majority of significant words (≥3 chars) from the concept name
                  appear in the chunk (handles models that paraphrase evidence)
    """
    if _normalize_ws(evidence) in chunk_norm:
        return True
    words = [w for w in _normalize_ws(concept_name).split() if len(w) >= 3]
    if not words:
        return False
    matches = sum(1 for w in words if w in chunk_norm)
    return matches >= max(1, len(words) - 1)


def extract_concepts_from_chunk(
    chunk: dict,
    transcript_segments: list,
    model: str,
) -> list[Concept]:
    """Extract and validate concepts from a single transcript chunk.

    Retries once with a JSON-only preamble if the first parse fails.
    Drops any concept whose evidence string cannot be found verbatim
    (case/whitespace-normalized) in the chunk text — these are hallucinations.
    """
    prompt = build_concept_prompt(chunk, transcript_segments)

    raw_concepts: list[dict] | None = None
    for attempt in range(2):
        actual_prompt = ("Return ONLY valid JSON. No prose.\n\n" + prompt) if attempt else prompt
        try:
            raw = llm.generate(actual_prompt, model=model, format_json=True)
            raw_concepts = parse_concept_response(raw)
            break
        except ConceptParseError as exc:
            if attempt == 0:
                logger.debug(
                    "Chunk %d: parse failed on first attempt, retrying: %s",
                    chunk["index"], exc,
                )
            else:
                logger.warning("Chunk %d: parse failed after retry", chunk["index"])
        # OllamaError intentionally not caught — propagates to orchestrator

    if raw_concepts is None:
        return []

    validated: list[Concept] = []
    hallucinated = 0
    chunk_norm = _normalize_ws(chunk["text"])

    for entry in raw_concepts:
        try:
            concept_name = str(entry.get("concept", "")).strip()
            if not concept_name or not (1 <= len(concept_name.split()) <= 6):
                continue

            explanation = str(entry.get("explanation", "")).strip()
            if not explanation:
                continue

            # Grounding check — concept name words must appear in chunk text
            evidence = str(entry.get("evidence", "")).strip()
            if not _is_grounded(concept_name, evidence, chunk_norm):
                hallucinated += 1
                logger.debug(
                    "Chunk %d: dropping '%s' — not grounded in chunk text",
                    chunk["index"], concept_name,
                )
                continue

            # Clamp timestamps to chunk bounds with ±2 s slack
            slack = 2.0
            t_min = chunk["start"] - slack
            t_max = chunk["end"] + slack
            ts_start = max(t_min, min(float(entry.get("timestamp_start", chunk["start"])), t_max))
            ts_end = max(t_min, min(float(entry.get("timestamp_end", chunk["end"])), t_max))
            if ts_start > ts_end:
                ts_start, ts_end = ts_end, ts_start

            keywords = [str(k).strip() for k in entry.get("keywords", []) if str(k).strip()]

            validated.append(Concept(
                concept=concept_name,
                keywords=keywords,
                explanation=explanation,
                evidence=evidence,
                timestamp_start=ts_start,
                timestamp_end=ts_end,
            ))
        except Exception as exc:
            logger.debug("Chunk %d: skipping malformed concept entry: %s", chunk["index"], exc)

    logger.debug(
        "Chunk %d: %d concepts validated, %d dropped (ungrounded evidence)",
        chunk["index"], len(validated), hallucinated,
    )
    return validated


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _normalize_concept_name(name: str) -> str:
    name = name.lower().strip()
    name = name.translate(str.maketrans("", "", string.punctuation))
    return " ".join(name.split())


def dedupe_concepts(concepts: list[Concept]) -> list[Concept]:
    """Merge duplicates from overlapping chunks using normalized name matching.

    Within each group of same-named concepts:
    - Keep the longest explanation
    - Expand timestamp range to cover all occurrences
    - Union all keywords
    """
    groups: dict[str, list[Concept]] = {}
    for c in concepts:
        key = _normalize_concept_name(c.concept)
        groups.setdefault(key, []).append(c)

    merged: list[Concept] = []
    for group in groups.values():
        best = max(group, key=lambda c: len(c.explanation))
        all_keywords = list({kw for c in group for kw in c.keywords})
        merged.append(Concept(
            concept=best.concept,
            keywords=all_keywords,
            explanation=best.explanation,
            evidence=best.evidence,
            timestamp_start=min(c.timestamp_start for c in group),
            timestamp_end=max(c.timestamp_end for c in group),
        ))

    return sorted(merged, key=lambda c: c.timestamp_start)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_concepts(
    url: str,
    model: str | None = None,
    force: bool = False,
) -> tuple[list[Concept], bool, int]:
    """Extract concepts for an already-transcribed video.

    Returns (concepts, from_cache, chunks_processed).

    Raises VideoNotReadyError, TranscriptNotReadyError, or
    ConceptExtractionFailedError. OllamaError propagates directly so
    the endpoint can map it to a 503.
    """
    video = state.get_video(url)
    if video is None:
        raise VideoNotReadyError(f"No ingested video for URL: {url}")

    if not video.get("transcript"):
        raise TranscriptNotReadyError(
            "No transcript found. Call POST /transcribe first."
        )

    if video.get("concepts") and not force:
        concepts = [Concept.model_validate(c) for c in video["concepts"]]
        return concepts, True, 0

    transcript = Transcript.model_validate(video["transcript"])
    chunks = chunk_transcript(transcript)
    _model = model or settings.OLLAMA_MODEL

    logger.info(
        "Concept extraction started: url=%s chunks=%d model=%s",
        url, len(chunks), _model,
    )
    t0 = time.monotonic()

    all_concepts: list[Concept] = []
    failed_chunks = 0

    for chunk in chunks:
        try:
            chunk_concepts = extract_concepts_from_chunk(
                chunk, transcript.segments, _model
            )
            all_concepts.extend(chunk_concepts)
            logger.debug(
                "Chunk %d/%d: %d concepts extracted",
                chunk["index"] + 1, len(chunks), len(chunk_concepts),
            )
        except OllamaError:
            raise  # fail fast — no point trying remaining chunks
        except Exception as exc:
            logger.warning(
                "Chunk %d/%d failed, continuing: %s",
                chunk["index"] + 1, len(chunks), exc,
            )
            failed_chunks += 1

    if failed_chunks == len(chunks) and chunks:
        raise ConceptExtractionFailedError(
            f"All {len(chunks)} chunk(s) failed during concept extraction."
        )

    merged = dedupe_concepts(all_concepts)

    video["concepts"] = [c.model_dump() for c in merged]

    elapsed = time.monotonic() - t0
    logger.info(
        "Concept extraction complete: %d concepts from %d chunks in %.1fs",
        len(merged), len(chunks), elapsed,
    )

    return merged, False, len(chunks)
