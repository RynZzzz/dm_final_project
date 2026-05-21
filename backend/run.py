import os
import uvicorn

# Tell HuggingFace to load from local cache only — prevents network calls
# that fail and crash the httpx client when there's no internet access.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
