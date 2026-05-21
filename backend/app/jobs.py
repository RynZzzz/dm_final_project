"""Job management helpers — thin wrappers around app.state."""

import uuid

from app import state


def create_new_job(url: str) -> str:
    """Create a new job entry and return its job_id."""
    job_id = str(uuid.uuid4())
    state.create_job(job_id, url)
    return job_id


def get_job_status(job_id: str) -> dict | None:
    """Return the job dict if it exists, else None."""
    return state.get_job(job_id)


def get_job_result(job_id: str) -> dict | None:
    """Return the result sub-dict for a completed job, else None."""
    job = state.get_job(job_id)
    if job is None:
        return None
    return job.get("result")


def delete_job(job_id: str) -> None:
    state.delete_job(job_id)


def delete_video(url: str) -> None:
    state.delete_video(url)
