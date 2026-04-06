"""
Ollama-based text polisher for cleaning up raw transcriptions.
"""

import json
import logging
import time
import requests

from core.style_learner import load_style

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

POLISH_PROMPT = """\
You are a text cleanup assistant. You receive raw speech-to-text transcription.

Your ONLY job is to clean up the transcription:
1. Remove filler words (um, uh, you know, like, so, okay), false starts, and repeated phrases.
2. Fix obvious mistranscriptions and grammar errors.
3. Restructure into clear, well-punctuated sentences and paragraphs.

CRITICAL RULES:
- NEVER add, invent, or fabricate information that is not in the original transcription.
- NEVER change the meaning or topic of what was said.
- If the input is casual or non-technical, keep it casual — just make it read cleanly.
- Do NOT wrap output in quotation marks.
- Do NOT add labels like "From your transcription:" or any other headers/prefixes.
- Output ONLY the cleaned-up text. No preamble, no commentary, no framing."""


def _build_system_prompt() -> str:
    """Build the full system prompt, injecting the user's writing style if available."""
    style = load_style()
    if style:
        return POLISH_PROMPT + "\n\nAdditionally, match the following writing style:\n" + style
    return POLISH_PROMPT


MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def polish_stream(raw_text: str):
    """Yield polished tokens from Ollama as they arrive.

    Yields dicts: ``{"token": str}`` for content, ``{"error": str}`` on
    failure, and ``{"done": True}`` when finished.
    Retries up to MAX_RETRIES times on 500 errors (usually GPU memory pressure).
    """
    token_count = 0
    t_start = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("Sending polish request to Ollama (%s) – %d chars of input (attempt %d/%d)",
                     OLLAMA_MODEL, len(raw_text), attempt, MAX_RETRIES)
            t_req = time.perf_counter()
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": raw_text,
                    "system": _build_system_prompt(),
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )

            # On 500, log the actual response body and retry
            if resp.status_code == 500:
                body = resp.text
                log.warning("Ollama returned 500 (attempt %d/%d): %s", attempt, MAX_RETRIES, body)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    yield {"error": f"Ollama keeps returning 500 after {MAX_RETRIES} attempts. Response: {body[:200]}"}
                    break

            resp.raise_for_status()
            log.info("Ollama responded (HTTP %s) in %.1fs – streaming tokens…",
                     resp.status_code, time.perf_counter() - t_req)
            t_first = None
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        if t_first is None:
                            t_first = time.perf_counter()
                            log.info("First token received %.1fs after request", t_first - t_req)
                        token_count += 1
                        yield {"token": token}
                    if chunk.get("done"):
                        break
            # Success – don't retry
            break
        except requests.ConnectionError:
            log.error("Cannot connect to Ollama at %s", OLLAMA_URL)
            yield {"error": "Cannot connect to Ollama. Is it running? (ollama serve)"}
            break
        except requests.HTTPError as e:
            log.error("Ollama HTTP error: %s", e)
            yield {"error": f"Ollama error: {e}. Have you run: ollama pull {OLLAMA_MODEL}?"}
            break
        except Exception as e:
            log.exception("Unexpected error during polish")
            yield {"error": str(e)}
            break

    log.info("Polish complete – %d tokens in %.1fs", token_count, time.perf_counter() - t_start)
    yield {"done": True}
