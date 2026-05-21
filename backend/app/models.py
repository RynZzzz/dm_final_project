from datetime import datetime

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    url: str


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

class TranscriptWord(BaseModel):
    start: float
    end: float
    word: str


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = []


class Transcript(BaseModel):
    language: str
    language_probability: float
    duration: float
    segments: list[TranscriptSegment]
    model_size: str
    device: str
    transcribed_at: datetime


class TranscribeRequest(BaseModel):
    url: str


class TranscribeResponse(BaseModel):
    url: str
    language: str
    duration: float
    segment_count: int
    word_count: int
    model_size: str
    device: str
    from_cache: bool


# ---------------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------------

class Concept(BaseModel):
    concept: str
    keywords: list[str]
    explanation: str
    evidence: str
    timestamp_start: float
    timestamp_end: float


class ExtractConceptsRequest(BaseModel):
    url: str


class ExtractConceptsResponse(BaseModel):
    url: str
    concept_count: int
    concepts: list[Concept]
    from_cache: bool
    chunks_processed: int
    model: str


# ---------------------------------------------------------------------------
# Comment classification
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    predicted_label: str
    confidence: float
    all_scores: dict[str, float]
    is_trouble: bool


class ClassifiedComment(BaseModel):
    comment_id: str
    author: str
    text: str
    timestamps_in_text: list[int]
    classification: ClassificationResult


class ClassifyRequest(BaseModel):
    url: str


class ClassifyResponse(BaseModel):
    url: str
    total_comments: int
    timestamped_comments: int
    classified_count: int
    trouble_count: int
    label_distribution: dict[str, int]
    from_cache: bool
    model: str
    threshold: float


# ---------------------------------------------------------------------------
# Concept–comment matching
# ---------------------------------------------------------------------------

class MatchedComment(BaseModel):
    comment_id: str
    author: str
    text: str
    timestamps_in_text: list[int]
    predicted_label: str
    classification_confidence: float
    similarity_score: float
    has_timestamp: bool = True


class ConceptMatch(BaseModel):
    concept: Concept
    matched_comments: list[MatchedComment]


class UnmatchedBucket(BaseModel):
    comments: list[MatchedComment]


class MatchResult(BaseModel):
    concept_matches: list[ConceptMatch]
    unmatched: UnmatchedBucket


class MatchRequest(BaseModel):
    url: str


class MatchResponse(BaseModel):
    url: str
    trouble_comments_total: int
    matched_count: int
    unmatched_count: int
    concepts_with_matches: int
    threshold: float
    from_cache: bool
    result: MatchResult


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

class CommentSample(BaseModel):
    text: str
    author: str = ""


class SentimentSummary(BaseModel):
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    sample_size: int
    disclaimer: str


class SentimentRequest(BaseModel):
    url: str


class SentimentResponse(BaseModel):
    url: str
    summary: SentimentSummary
    from_cache: bool
    model: str


# ---------------------------------------------------------------------------
# Concept synthesis
# ---------------------------------------------------------------------------

class ConceptSynthesis(BaseModel):
    concept_name: str
    summary: str
    quoted_evidence: list[str]
    matched_comment_count: int


class SynthesisRequest(BaseModel):
    url: str


class SynthesisResponse(BaseModel):
    url: str
    syntheses: list[ConceptSynthesis]
    concepts_synthesized: int
    concepts_skipped: int
    from_cache: bool
    model: str


# ---------------------------------------------------------------------------
# Orchestration / async job
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    video_id: str
    title: str
    uploader: str
    duration: float
    view_count: int | None = None
    upload_date: str | None = None


DISCLAIMERS = [
    "All comments are analyzed. Comments with explicit timestamps (e.g. '3:45') "
    "are matched to concepts at a lower similarity threshold (0.40). Comments "
    "without timestamps are matched by content alone at a stricter threshold (0.55).",
    "Zero-shot classification has no fine-tuning on educational content. "
    "Labels are probabilistic and may misclassify ambiguous comments.",
    "Sentiment analysis reflects the tone of comments that exist on the video, "
    "not all viewers. YouTube comment sections tend to skew positive.",
]


class ConceptReport(BaseModel):
    concept_name: str
    keywords: list[str]
    explanation: str
    timestamp_start: float
    timestamp_end: float
    matched_comment_count: int
    matched_comments: list[MatchedComment] = []
    synthesis: ConceptSynthesis | None = None


class AnalysisResult(BaseModel):
    url: str
    metadata: VideoMetadata
    total_comments: int
    timestamped_comments: int
    trouble_comment_count: int
    unmatched_trouble_count: int
    unmatched_trouble_comments: list[MatchedComment] = []
    concept_reports: list[ConceptReport]
    sentiment: SentimentSummary | None = None
    label_distribution: dict[str, int] = {}
    label_comment_samples: dict[str, list[CommentSample]] = {}
    sentiment_comment_samples: dict[str, list[CommentSample]] = {}
    disclaimers: list[str] = []


class JobStatus(BaseModel):
    job_id: str
    status: str          # pending | running | completed | failed
    stage: str | None
    progress: int        # 0–100
    video_url: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class AnalyzeRequest(BaseModel):
    url: str
    force: bool = False


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str
    video_url: str
