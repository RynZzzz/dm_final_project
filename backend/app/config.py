from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    YOUTUBE_API_KEY: str = ""
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi4-mini"
    WHISPER_MODEL_SIZE: str = "base"
    LOG_LEVEL: str = "INFO"
    # Zero-shot classifier model (HuggingFace model ID)
    CLASSIFIER_MODEL: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    # Minimum confidence for a comment to be flagged as "trouble"
    CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.50
    # Minimum cosine similarity for a comment to be assigned to a concept
    SIMILARITY_THRESHOLD: float = 0.40

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
