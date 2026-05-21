"""
In-memory state store. All state is ephemeral and lost on server restart.
There is no database and no disk persistence — this is intentional.

Eviction: we keep at most MAX_VIDEOS video entries and MAX_JOBS job entries.
When the cap is hit the oldest entry (by insertion order) is dropped.
"""

from datetime import datetime, timezone

MAX_VIDEOS = 10
MAX_JOBS   = 50

jobs:   dict[str, dict] = {}
videos: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def _evict(d: dict, max_size: int) -> None:
    while len(d) > max_size:
        oldest = next(iter(d))
        del d[oldest]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def get_job(job_id: str) -> dict | None:
    return jobs.get(job_id)

# alias used by main.py
get_job_status = get_job


def create_job(job_id: str, video_url: str) -> dict:
    now = datetime.now(timezone.utc)
    job = {
        "status": "pending",
        "stage": None,
        "progress": 0,
        "video_url": video_url,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
        "result": None,
    }
    jobs[job_id] = job
    _evict(jobs, MAX_JOBS)
    return job

# alias used by main.py
def create_new_job(video_url: str) -> str:
    import uuid
    job_id = str(uuid.uuid4())
    create_job(job_id, video_url)
    return job_id


def update_job(job_id: str, **fields) -> dict:
    jobs[job_id].update(fields)
    jobs[job_id]["updated_at"] = datetime.now(timezone.utc)
    return jobs[job_id]


def get_job_result(job_id: str) -> dict | None:
    job = jobs.get(job_id)
    if job is None:
        return None
    return job.get("result")


def delete_job(job_id: str) -> None:
    jobs.pop(job_id, None)


def delete_video(url: str) -> None:
    videos.pop(url, None)


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

def get_video(url: str) -> dict | None:
    return videos.get(url)


def set_video(url: str, data: dict) -> None:
    videos[url] = data
    _evict(videos, MAX_VIDEOS)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats() -> dict:
    return {
        "jobs_tracked":   len(jobs),
        "videos_cached":  len(videos),
    }
