import json
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    pass

class OllamaUnreachableError(OllamaError):
    pass

class OllamaTimeoutError(OllamaError):
    pass

class OllamaInvalidResponseError(OllamaError):
    pass


def generate(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    format_json: bool = False,
    token_timeout: float = 60.0,
) -> str:
    """Call Ollama's /api/generate (streaming) and return the full response text.

    Uses streaming so there is no global wall-clock timeout — the model can
    take as long as it needs.  token_timeout is the maximum seconds allowed
    between consecutive tokens; if Ollama goes silent longer than that the
    call is aborted.

    Raises OllamaUnreachableError, OllamaTimeoutError, or
    OllamaInvalidResponseError on failure.
    """
    _model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_HOST}/api/generate"

    body: dict = {
        "model": _model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": 0,  # unload from VRAM immediately after response
        "options": {
            "temperature": temperature,
            "num_ctx": 2048,
            "num_predict": 300,
        },
    }
    if format_json:
        body["format"] = "json"

    logger.debug(
        "Ollama generate: model=%s temperature=%.2f prompt_len=%d",
        _model, temperature, len(prompt),
    )

    # connect timeout: 10 s; read timeout (per token batch): token_timeout
    timeouts = httpx.Timeout(connect=10.0, read=token_timeout, write=10.0, pool=5.0)

    try:
        parts: list[str] = []
        with httpx.stream("POST", url, json=body, timeout=timeouts) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                parts.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break

        result = "".join(parts)
        logger.debug("Ollama response: length=%d", len(result))
        return result

    except httpx.ConnectError as exc:
        raise OllamaUnreachableError(
            f"Cannot connect to Ollama at {settings.OLLAMA_HOST}. "
            "Ensure 'ollama serve' is running."
        ) from exc
    except httpx.ReadTimeout as exc:
        logger.warning("Ollama token timeout after %.0fs of silence", token_timeout)
        raise OllamaTimeoutError(
            f"Ollama stopped producing tokens for {token_timeout:.0f}s (model may be overloaded)."
        ) from exc
    except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise OllamaInvalidResponseError(
            f"Unexpected response from Ollama: {exc}"
        ) from exc
