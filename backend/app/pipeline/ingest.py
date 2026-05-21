import dataclasses
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app import state
from app.config import settings
from app.pipeline import preprocess

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IngestError(Exception):
    pass


class VideoUnavailableError(IngestError):
    def __init__(self, url: str):
        super().__init__(f"Video unavailable: {url}")


class AgeRestrictedError(IngestError):
    def __init__(self, url: str):
        super().__init__(f"Video is age-restricted: {url}")


class CommentsDisabledError(IngestError):
    def __init__(self, video_id: str):
        super().__init__(f"Comments are disabled for video: {video_id}")


class QuotaExceededError(IngestError):
    def __init__(self):
        super().__init__("YouTube Data API quota exceeded")


class GenericIngestError(IngestError):
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Comment:
    id: str
    text: str
    author: str
    like_count: int
    published_at: str
    timestamps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(url: str) -> tuple[Path, dict]:
    """Download best audio from a YouTube URL, convert to 16 kHz mono WAV.

    Returns the path to the WAV file and a dict of video metadata.
    The caller is responsible for cleaning up the temp directory.
    """
    tmpdir = Path(tempfile.mkdtemp())

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        # -ar 16000 -ac 1: 16 kHz mono, required by Whisper
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        "ffmpeg_location": r"Z:\Program\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        msg = str(exc).lower()
        if "age" in msg and ("restrict" in msg or "confirm" in msg):
            raise AgeRestrictedError(url) from exc
        if any(p in msg for p in ("video unavailable", "not available", "private video")):
            raise VideoUnavailableError(url) from exc
        raise GenericIngestError(str(exc)) from exc

    # yt-dlp may return a playlist wrapper; unwrap single-video playlists
    if "entries" in info:
        info = info["entries"][0]

    wav_files = list(tmpdir.glob("*.wav"))
    if not wav_files:
        raise GenericIngestError(f"No WAV file produced in {tmpdir}")
    wav_path = wav_files[0]

    video_info = {
        "id": info.get("id"),
        "title": info.get("title"),
        "description": info.get("description"),
        "duration": info.get("duration"),          # seconds
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),    # YYYYMMDD string
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "channel_id": info.get("channel_id"),
    }

    logger.info("Downloaded audio for video %s → %s", video_info["id"], wav_path)
    return wav_path, video_info


# ---------------------------------------------------------------------------
# Comment fetching
# ---------------------------------------------------------------------------

def fetch_comments(
    video_id: str,
    api_key: str,
    max_comments: int = 1000,
) -> list[Comment]:
    """Fetch top-level comments via YouTube Data API v3.

    Paginates until max_comments is reached or no further pages exist.
    Each comment is cleaned and scanned for embedded timestamps.
    """
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
    except Exception as exc:
        raise GenericIngestError(f"Failed to initialise YouTube client: {exc}") from exc

    comments: list[Comment] = []
    page_token: str | None = None

    while len(comments) < max_comments:
        page_size = min(100, max_comments - len(comments))

        try:
            response = (
                youtube.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=page_size,
                    order="relevance",
                    pageToken=page_token,
                )
                .execute()
            )
        except HttpError as exc:
            _classify_api_error(exc, video_id)

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = preprocess.clean_comment_text(snippet.get("textDisplay", ""))
            comments.append(Comment(
                id=item["id"],
                text=text,
                author=snippet.get("authorDisplayName", ""),
                like_count=snippet.get("likeCount", 0),
                published_at=snippet.get("publishedAt", ""),
                timestamps=preprocess.extract_timestamps(text),
            ))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    logger.info("Fetched %d comments for video %s", len(comments), video_id)
    return comments


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _serialize_comments(comments: list[Comment]) -> list[dict]:
    """Convert Comment dataclasses to state-storable dicts with derived fields."""
    def _ts_to_seconds(ts: str) -> int:
        try:
            parts = [int(p) for p in ts.split(":")]
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
        except (ValueError, IndexError):
            pass
        return 0

    result = []
    for c in comments:
        d = dataclasses.asdict(c)
        ts_seconds = [_ts_to_seconds(t) for t in d.get("timestamps", [])]
        d["timestamps_seconds"] = ts_seconds
        d["has_timestamp"] = len(ts_seconds) > 0
        result.append(d)
    return result


def ingest_video(url: str) -> dict:
    """Download audio, fetch comments, and store everything in state.

    Returns the video info dict stored in state.
    Raises IngestError (or a subclass) on any failure.
    """
    wav_path, video_info = download_audio(url)

    comments_data: list[dict] = []
    if settings.YOUTUBE_API_KEY and video_info.get("id"):
        try:
            raw = fetch_comments(video_info["id"], settings.YOUTUBE_API_KEY)
            comments_data = _serialize_comments(raw)
            logger.info("Fetched %d comments for %s", len(comments_data), video_info["id"])
        except Exception as exc:
            logger.warning("Comment fetch failed (continuing without comments): %s", exc)

    data = {
        **video_info,
        "audio_path": str(wav_path),
        "transcript": None,
        "comments": comments_data,
    }
    state.set_video(url, data)
    logger.info("Ingested video %s → %s", video_info.get("id"), wav_path)
    return data


def _classify_api_error(exc: HttpError, video_id: str) -> None:
    status = int(exc.resp.status)
    reason = ""
    try:
        body = json.loads(exc.content)
        errors = body.get("error", {}).get("errors", [])
        reason = errors[0].get("reason", "") if errors else ""
    except Exception:
        pass

    if status == 403:
        if reason in ("quotaExceeded", "dailyLimitExceeded"):
            raise QuotaExceededError() from exc
        # commentsDisabled, forbidden, etc.
        raise CommentsDisabledError(video_id) from exc
    if status == 404:
        raise VideoUnavailableError(video_id) from exc
    raise GenericIngestError(str(exc)) from exc
