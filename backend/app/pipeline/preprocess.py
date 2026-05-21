import html
import re

# Matches MM:SS and HH:MM:SS timestamp formats (e.g. "4:20", "1:04:30")
_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def clean_comment_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_timestamps(text: str) -> list[str]:
    return _TIMESTAMP_RE.findall(text)
