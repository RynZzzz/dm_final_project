import dataclasses
import logging
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app import state
from app.models import (
    IngestRequest,
    TranscribeRequest,
    TranscribeResponse,
    ExtractConceptsRequest,
    ExtractConceptsResponse,
    ClassifyRequest,
    ClassifyResponse,
    MatchRequest,
    MatchResponse,
    SentimentRequest,
    SentimentResponse,
    SynthesisRequest,
    SynthesisResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    JobStatus,
    AnalysisResult,
)
from app.pipeline.ingest import (
    download_audio,
    fetch_comments,
    VideoUnavailableError,
    AgeRestrictedError,
    GenericIngestError,
)
from app.pipeline import orchestrator
from app import jobs
from app.pipeline.transcribe import (
    transcribe_video,
    VideoNotIngestedError,
    AudioNotAvailableError,
    TranscriptionFailedError,
    TranscribeError,
)
from app.pipeline.concepts import (
    extract_concepts,
    VideoNotReadyError,
    TranscriptNotReadyError,
    ConceptExtractionFailedError,
)
from app.pipeline.llm import OllamaUnreachableError
from app.pipeline.classify import (
    classify_video_comments,
    compute_label_distribution,
    VideoNotReadyError as ClassifyVideoNotReadyError,
    CommentsNotReadyError,
    ModelLoadError,
    ClassificationError,
)
from app.pipeline.match import (
    match_video,
    VideoNotReadyError as MatchVideoNotReadyError,
    ConceptsNotReadyError,
    ClassifiedCommentsNotReadyError,
    EmbeddingFailedError,
    MatchingError,
)
from app.pipeline.sentiment import (
    analyze_video_sentiment,
    VideoNotReadyError as SentimentVideoNotReadyError,
    CommentsNotReadyError as SentimentCommentsNotReadyError,
    SentimentError,
    MODEL_NAME as SENTIMENT_MODEL_NAME,
)
from app.pipeline.synthesize import (
    synthesize_video,
    VideoNotReadyError as SynthesisVideoNotReadyError,
    TranscriptNotReadyError as SynthesisTranscriptNotReadyError,
    MatchesNotReadyError,
    SynthesisError,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="YT Analysis Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    logger.info("Server running in in-memory mode — all state is ephemeral")


@app.get("/")
async def root():
    return {"message": "YT Analysis Backend is running"}


@app.get("/health")
async def health():
    counts = state.stats()
    return {
        "status": "ok",
        "version": "0.1.0",
        "state": "in-memory",
        "jobs_tracked": counts["jobs_tracked"],
        "videos_cached": counts["videos_cached"],
    }


def _ts_string_to_seconds(ts: str) -> int:
    """Convert 'MM:SS' or 'HH:MM:SS' timestamp string to total seconds."""
    try:
        parts = [int(p) for p in ts.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except (ValueError, IndexError):
        pass
    return 0


def _serialize_comments(comments) -> list[dict]:
    """Convert Comment dataclass objects to state-storable dicts.

    Adds has_timestamp (bool) and timestamps_seconds (list[int]) derived
    fields so the classify stage can read them without re-parsing.
    """
    result = []
    for c in comments:
        d = dataclasses.asdict(c)
        ts_seconds = [_ts_string_to_seconds(t) for t in d.get("timestamps", [])]
        d["timestamps_seconds"] = ts_seconds
        d["has_timestamp"] = len(ts_seconds) > 0
        result.append(d)
    return result


@app.post("/ingest")
def ingest_endpoint(request: IngestRequest):
    """Download and cache audio for a YouTube URL.

    Stores the audio path and video metadata in state. The audio is
    consumed (and deleted) by POST /transcribe.
    """
    try:
        wav_path, video_info = download_audio(request.url)
    except VideoUnavailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except AgeRestrictedError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except GenericIngestError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Fetch comments when API key is configured; store empty list otherwise.
    comments_data: list[dict] = []
    if settings.YOUTUBE_API_KEY and video_info.get("id"):
        try:
            raw = fetch_comments(video_info["id"], settings.YOUTUBE_API_KEY)
            comments_data = _serialize_comments(raw)
            logger.info("Fetched %d comments for %s", len(comments_data), video_info["id"])
        except Exception as exc:
            logger.warning("Comment fetch failed (continuing without comments): %s", exc)

    state.set_video(request.url, {
        **video_info,
        "audio_path": str(wav_path),
        "transcript": None,
        "comments": comments_data,
    })

    logger.info("Ingested video %s → %s", video_info.get("id"), wav_path)

    return {
        "video_id": video_info.get("id"),
        "title": video_info.get("title"),
        "duration": video_info.get("duration"),
        "audio_path": str(wav_path),
        "comments_fetched": len(comments_data),
    }


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_endpoint(request: TranscribeRequest, force: bool = False):
    """Transcribe an already-ingested video.

    The video must have been ingested first via POST /ingest.
    Pass ?force=true to re-transcribe even if a cached transcript exists
    (requires re-ingesting first, since the audio is deleted on first
    successful transcription).
    """
    try:
        transcript, from_cache = transcribe_video(
            url=request.url,
            model_size=settings.WHISPER_MODEL_SIZE,
            force=force,
        )
    except VideoNotIngestedError:
        raise HTTPException(
            status_code=404,
            detail="Video has not been ingested. Call POST /ingest first.",
        )
    except AudioNotAvailableError:
        raise HTTPException(
            status_code=409,
            detail="Audio file is no longer available. Re-ingest the video with POST /ingest.",
        )
    except TranscriptionFailedError as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    except TranscribeError as exc:
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")

    word_count = sum(len(s.words) for s in transcript.segments)

    return TranscribeResponse(
        url=request.url,
        language=transcript.language,
        duration=transcript.duration,
        segment_count=len(transcript.segments),
        word_count=word_count,
        model_size=transcript.model_size,
        device=transcript.device,
        from_cache=from_cache,
    )


@app.post("/extract-concepts", response_model=ExtractConceptsResponse)
def extract_concepts_endpoint(
    request: ExtractConceptsRequest,
    force: bool = False,
):
    """Extract concepts from an already-transcribed video.

    The video must have been ingested AND transcribed first.
    Pass ?force=true to re-extract even if cached concepts exist.
    """
    try:
        concepts, from_cache, chunks = extract_concepts(
            url=request.url,
            model=settings.OLLAMA_MODEL,
            force=force,
        )
    except VideoNotReadyError:
        raise HTTPException(
            status_code=404,
            detail="Video has not been ingested. Call POST /ingest first.",
        )
    except TranscriptNotReadyError:
        raise HTTPException(
            status_code=409,
            detail="Video has no transcript. Call POST /transcribe first.",
        )
    except OllamaUnreachableError:
        raise HTTPException(
            status_code=503,
            detail="Ollama server is unreachable. Ensure 'ollama serve' is running.",
        )
    except ConceptExtractionFailedError as exc:
        raise HTTPException(status_code=500, detail=f"Concept extraction failed: {exc}")

    return ExtractConceptsResponse(
        url=request.url,
        concept_count=len(concepts),
        concepts=concepts,
        from_cache=from_cache,
        chunks_processed=chunks,
        model=settings.OLLAMA_MODEL,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(request: ClassifyRequest, force: bool = False):
    """Classify the timestamped comments of an ingested video.

    Transcription and concept extraction are not required — this endpoint
    is independent and can run after /ingest alone.
    Pass ?force=true to re-classify even if a cached result exists.
    """
    try:
        classified, from_cache = classify_video_comments(
            url=request.url,
            threshold=settings.CLASSIFIER_CONFIDENCE_THRESHOLD,
            force=force,
        )
    except ClassifyVideoNotReadyError:
        raise HTTPException(
            status_code=404,
            detail="Video has not been ingested. Call POST /ingest first.",
        )
    except CommentsNotReadyError:
        raise HTTPException(
            status_code=409,
            detail="Video has no comments. Ensure YOUTUBE_API_KEY is set and re-ingest.",
        )
    except ModelLoadError as exc:
        raise HTTPException(status_code=500, detail=f"Classifier model failed to load: {exc}")
    except ClassificationError as exc:
        raise HTTPException(status_code=500, detail=f"Classification error: {exc}")

    video = state.get_video(request.url)
    all_comments: list[dict] = video.get("comments", [])
    total = len(all_comments)
    timestamped = sum(1 for c in all_comments if c.get("has_timestamp"))
    trouble = sum(1 for c in classified if c.classification.is_trouble)
    distribution = compute_label_distribution(classified)

    return ClassifyResponse(
        url=request.url,
        total_comments=total,
        timestamped_comments=timestamped,
        classified_count=len(classified),
        trouble_count=trouble,
        label_distribution=distribution,
        from_cache=from_cache,
        model=settings.CLASSIFIER_MODEL,
        threshold=settings.CLASSIFIER_CONFIDENCE_THRESHOLD,
    )


@app.post("/match", response_model=MatchResponse)
def match_endpoint(request: MatchRequest, force: bool = False):
    """Match trouble comments to concepts via cosine similarity.

    Requires the video to have been ingested, had concepts extracted
    (POST /extract-concepts), and had comments classified (POST /classify).
    Pass ?force=true to re-match even if a cached result exists.
    """
    try:
        result, from_cache = match_video(
            url=request.url,
            threshold=settings.SIMILARITY_THRESHOLD,
            force=force,
        )
    except MatchVideoNotReadyError:
        raise HTTPException(status_code=404, detail="Video has not been ingested.")
    except ConceptsNotReadyError:
        raise HTTPException(
            status_code=409,
            detail="Concepts have not been extracted. Call POST /extract-concepts first.",
        )
    except ClassifiedCommentsNotReadyError:
        raise HTTPException(
            status_code=409,
            detail="Comments have not been classified. Call POST /classify first.",
        )
    except EmbeddingFailedError as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")
    except MatchingError as exc:
        raise HTTPException(status_code=500, detail=f"Matching error: {exc}")

    matched_count = sum(len(cm.matched_comments) for cm in result.concept_matches)
    unmatched_count = len(result.unmatched.comments)
    concepts_with_matches = sum(1 for cm in result.concept_matches if cm.matched_comments)

    return MatchResponse(
        url=request.url,
        trouble_comments_total=matched_count + unmatched_count,
        matched_count=matched_count,
        unmatched_count=unmatched_count,
        concepts_with_matches=concepts_with_matches,
        threshold=settings.SIMILARITY_THRESHOLD,
        from_cache=from_cache,
        result=result,
    )


@app.post("/sentiment", response_model=SentimentResponse)
def sentiment_endpoint(request: SentimentRequest, force: bool = False):
    """Run aggregate sentiment analysis on all comments for an ingested video.

    Independent of transcription and concept extraction; can run after
    /ingest alone. Pass ?force=true to recompute.
    """
    try:
        summary, from_cache = analyze_video_sentiment(url=request.url, force=force)
    except SentimentVideoNotReadyError:
        raise HTTPException(status_code=404, detail="Video has not been ingested.")
    except SentimentCommentsNotReadyError:
        raise HTTPException(
            status_code=409,
            detail="Video has no comments. Ensure YOUTUBE_API_KEY is set and re-ingest.",
        )
    except SentimentError as exc:
        raise HTTPException(status_code=500, detail=f"Sentiment error: {exc}")

    return SentimentResponse(
        url=request.url,
        summary=summary,
        from_cache=from_cache,
        model=SENTIMENT_MODEL_NAME,
    )


@app.post("/synthesize", response_model=SynthesisResponse)
def synthesize_endpoint(request: SynthesisRequest, force: bool = False):
    """Generate per-concept synthesis summaries for a fully analyzed video.

    Requires ingest, transcribe, extract-concepts, classify, and match to
    have all run. Pass ?force=true to regenerate.
    """
    try:
        syntheses, skipped, from_cache = synthesize_video(
            url=request.url,
            model=settings.OLLAMA_MODEL,
            force=force,
        )
    except SynthesisVideoNotReadyError:
        raise HTTPException(status_code=404, detail="Video has not been ingested.")
    except SynthesisTranscriptNotReadyError:
        raise HTTPException(
            status_code=409,
            detail="Transcript not available. Call POST /transcribe first.",
        )
    except MatchesNotReadyError:
        raise HTTPException(
            status_code=409,
            detail="Match results not available. Call POST /match first.",
        )
    except OllamaUnreachableError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is unreachable. Ensure 'ollama serve' is running.",
        )
    except SynthesisError as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis error: {exc}")

    return SynthesisResponse(
        url=request.url,
        syntheses=syntheses,
        concepts_synthesized=len(syntheses),
        concepts_skipped=skipped,
        from_cache=from_cache,
        model=settings.OLLAMA_MODEL,
    )


@app.post("/analyze", response_model=AnalyzeResponse, status_code=202)
def analyze_endpoint(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Kick off the full analysis pipeline as a background job.

    Returns immediately with a job_id. Poll GET /status/{job_id} for progress
    and GET /results/{job_id} when the job completes.
    """
    job_id = jobs.create_new_job(request.url)
    background_tasks.add_task(orchestrator.run_pipeline, job_id, request.url, request.force)
    logger.info("Enqueued analysis job %s for %s", job_id, request.url)
    return AnalyzeResponse(job_id=job_id, status="pending", video_url=request.url)


@app.get("/status/{job_id}", response_model=JobStatus)
def status_endpoint(job_id: str):
    """Return the current status and progress of an analysis job."""
    job = jobs.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        stage=job.get("stage"),
        progress=job.get("progress", 0),
        video_url=job["video_url"],
        error_message=job.get("error_message"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@app.get("/results/{job_id}", response_model=AnalysisResult)
def results_endpoint(job_id: str):
    """Return the full analysis result for a completed job.

    Clears both the job and its video from memory after serving —
    results are single-use; call /reset/{job_id} to clean up early.
    Returns 404 if the job does not exist, 409 if it has not yet completed.
    """
    job = jobs.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not yet completed (status={job['status']!r}).",
        )
    result = jobs.get_job_result(job_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Job marked complete but result is missing.")

    validated = AnalysisResult.model_validate(result)

    # Free memory now that the result has been delivered
    video_url = job.get("video_url")
    jobs.delete_job(job_id)
    if video_url:
        jobs.delete_video(video_url)

    return validated


@app.delete("/reset/{job_id}", status_code=204)
def reset_endpoint(job_id: str):
    """Explicitly clear a job and its associated video from memory.

    Called by the frontend Reset button. Safe to call at any time —
    silently does nothing if the job is already gone.
    """
    job = jobs.get_job_status(job_id)
    video_url = job.get("video_url") if job else None
    jobs.delete_job(job_id)
    if video_url:
        jobs.delete_video(video_url)


# ---------------------------------------------------------------------------
# Serve React frontend (must be registered AFTER all API routes)
# ---------------------------------------------------------------------------

_DIST = Path(__file__).resolve().parent.parent.parent / "frontend-react" / "dist"

if _DIST.exists():
    _ASSETS = _DIST / "assets"
    if _ASSETS.exists():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve the React SPA for any route not matched by the API."""
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(_DIST / "index.html"))
else:
    logger.warning(
        "React build not found at %s — run 'npm run build' inside frontend-react/",
        _DIST,
    )
