# YT Analysis Backend

A FastAPI backend for analyzing YouTube videos — extracting transcripts,
identifying key concepts, and classifying content using local ML models.
Built in phases; currently at Phase 2 (transcription).

## In-Memory Design

All state is stored in module-level Python dicts in `app/state.py`. There is
no database and no disk persistence. State resets completely on server restart.
This is an intentional design choice for simplicity during development.

## Prerequisites

- Python 3.11+
- ffmpeg on PATH (required by yt-dlp for audio conversion)

## Setup

1. Clone the repo
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the example env file and fill in any keys:
   ```bash
   cp .env.example .env
   ```

## Running

```bash
./run.sh
```

Or directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Verify

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "state": "in-memory",
  "jobs_tracked": 0,
  "videos_cached": 0
}
```

## Usage

### 1. Ingest a video

Downloads the audio and caches it for transcription.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=<VIDEO_ID>"}'
```

### 2. Transcribe

```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=<VIDEO_ID>"}'
```

Response:
```json
{
  "url": "https://www.youtube.com/watch?v=...",
  "language": "en",
  "duration": 342.5,
  "segment_count": 87,
  "word_count": 1204,
  "model_size": "base",
  "device": "cpu",
  "from_cache": false
}
```

Call the same endpoint again — the response is instant and `from_cache` is `true`.

To force re-transcription (requires re-ingesting first):
```bash
curl -X POST "http://localhost:8000/transcribe?force=true" \
  -H "Content-Type: application/json" \
  -d '{"url": "..."}'
```

## Transcription Notes

- Uses [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
  (CTranslate2-based; significantly faster than openai-whisper)
- Model size is set by `WHISPER_MODEL_SIZE` in `.env` (default: `base`)
- First run downloads model weights to `~/.cache/huggingface/`:
  - `base` — ~75 MB
  - `small` — ~250 MB
  - `medium` — ~770 MB
- Device auto-detected: CUDA if available, CPU otherwise
- On CPU with the `base` model: a 10-minute video takes ~1–3 minutes
- On GPU: significantly faster
- Transcripts include word-level timestamps for downstream fine-grained matching
- Transcripts are cached in memory; restarting the server loses them
- The audio file is deleted immediately after successful transcription;
  re-ingest to transcribe again

## Orchestrated pipeline (Phase 7+)

Submit a URL and get back a `job_id` immediately. The full pipeline runs in
the background — ingest → transcribe → extract concepts → classify comments →
match → sentiment → synthesize.

```bash
# Submit
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=<VIDEO_ID>"}'
# → {"job_id": "...", "status": "pending", "video_url": "..."}

# Poll progress
curl http://localhost:8000/status/<job_id>
# → {"status": "running", "stage": "transcribe", "progress": 25, ...}

# Fetch result when completed
curl http://localhost:8000/results/<job_id>
```

Pass `"force": true` in the `/analyze` body to bypass all caches.

## Frontend

A Streamlit UI lives in `../frontend/`. See the frontend README or run:

```bash
cd ../frontend
pip install -r requirements.txt
cp .env.example .env
./run.sh
```

Navigate to http://localhost:8501 — enter a YouTube URL and watch the
pipeline run. Results are displayed concept-by-concept with matched
comments, syntheses, and sentiment.

## Phase Status

| Phase | Description                                   | Status |
|-------|-----------------------------------------------|--------|
| 0     | Scaffolding                                   | Done   |
| 1     | Ingest (audio + comments)                     | Done   |
| 2     | Transcription (Whisper)                       | Done   |
| 3     | Concept extraction (Ollama)                   | Done   |
| 4     | Comment classification (DeBERTa zero-shot)    | Done   |
| 5     | Embeddings + matching (sentence-transformers) | Done   |
| 6     | Sentiment analysis + synthesis                | Done   |
| 7     | Orchestration + async jobs                    | Done   |
| 8     | Streamlit frontend                            | Done   |
