"""
Faster-Whisper transcription. Consumes the audio file produced by the
ingest stage and produces a timestamped transcript. Always cleans up
the audio file in a finally block.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from faster_whisper import WhisperModel

from app import state
from app.models import Transcript, TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TranscribeError(Exception):
    pass

class VideoNotIngestedError(TranscribeError):
    pass

class AudioNotAvailableError(TranscribeError):
    pass

class TranscriptionFailedError(TranscribeError):
    pass


# ---------------------------------------------------------------------------
# Model cache — keyed by model_size so different sizes can coexist
# ---------------------------------------------------------------------------

# Each entry: {"model": WhisperModel, "device": str, "compute_type": str}
_model_cache: dict[str, dict] = {}


def get_model(model_size: str) -> tuple[WhisperModel, str, str]:
    """Return a cached Faster-Whisper model instance plus device/compute_type.

    Loads and caches the model on first call for a given size.
    """
    if model_size in _model_cache:
        entry = _model_cache[model_size]
        return entry["model"], entry["device"], entry["compute_type"]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        "Loading Whisper model '%s' on %s (compute_type=%s)",
        model_size, device, compute_type,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("Whisper model '%s' ready", model_size)

    _model_cache[model_size] = {
        "model": model,
        "device": device,
        "compute_type": compute_type,
    }
    return model, device, compute_type


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_path: Path, model_size: str) -> Transcript:
    """Transcribe an audio file. Returns a Transcript Pydantic object.

    Does not delete the audio file — the caller (transcribe_video) owns
    cleanup via a try/finally block.
    """
    model, device, _ = get_model(model_size)

    logger.info("Transcription started: %s", audio_path.name)
    t0 = time.monotonic()

    segments_iter, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        beam_size=5,
    )

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:  # consume the generator eagerly
        words: list[TranscriptWord] = []
        if seg.words:
            for w in seg.words:
                words.append(TranscriptWord(
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    word=w.word,
                ))
        segments.append(TranscriptSegment(
            start=round(seg.start, 3),
            end=round(seg.end, 3),
            text=seg.text.strip(),
            words=words,
        ))

    elapsed = time.monotonic() - t0
    logger.info(
        "Transcription complete: language=%s (%.0f%%), %d segments, "
        "duration=%.1fs, wall_time=%.1fs",
        info.language,
        info.language_probability * 100,
        len(segments),
        info.duration,
        elapsed,
    )

    return Transcript(
        language=info.language,
        language_probability=round(info.language_probability, 4),
        duration=round(info.duration, 2),
        segments=segments,
        model_size=model_size,
        device=device,
        transcribed_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def transcribe_video(
    url: str,
    model_size: str,
    force: bool = False,
) -> tuple[Transcript, bool]:
    """Transcribe a video that has already been ingested.

    Returns (Transcript, from_cache). Raises:
      VideoNotIngestedError  — URL not found in state
      AudioNotAvailableError — audio_path missing or file deleted
      TranscriptionFailedError — Whisper threw an unexpected error

    The audio file is always cleaned up in a finally block, even on
    failure, to avoid leaking temp files.
    """
    video = state.get_video(url)
    if video is None:
        raise VideoNotIngestedError(f"No ingested video for URL: {url}")

    # Return cached transcript when available and not forcing a redo
    if video.get("transcript") and not force:
        return Transcript.model_validate(video["transcript"]), True

    audio_path_str = video.get("audio_path")
    if not audio_path_str:
        raise AudioNotAvailableError(
            "Audio file has already been deleted after a previous transcription. "
            "Re-ingest the video with POST /ingest."
        )

    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        raise AudioNotAvailableError(
            f"Audio file not found at {audio_path}. "
            "Re-ingest the video with POST /ingest."
        )

    transcript: Transcript | None = None
    try:
        transcript = transcribe_audio(audio_path, model_size)
    except (VideoNotIngestedError, AudioNotAvailableError):
        raise
    except Exception as exc:
        raise TranscriptionFailedError(str(exc)) from exc
    finally:
        # Always clean up — on success audio_path becomes stale, on failure
        # we still don't want to leave multi-hundred-MB WAV files behind.
        _cleanup_audio(audio_path)
        video["audio_path"] = None

    video["transcript"] = transcript.model_dump(mode="json")

    return transcript, False


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------

def _cleanup_audio(audio_path: Path) -> None:
    """Delete the audio file and its (empty) temp dir. Never raises."""
    try:
        if audio_path.exists():
            audio_path.unlink()
        try:
            audio_path.parent.rmdir()  # only succeeds if dir is empty
        except OSError:
            pass
    except Exception as exc:
        logger.warning("Could not clean up audio file %s: %s", audio_path, exc)
