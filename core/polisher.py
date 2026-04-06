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
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"

POLISH_SYSTEM = """\
You clean up raw speech-to-text transcription. You fix grammar, remove filler words, and make it read clearly.

ABSOLUTE CONSTRAINTS:
- The output must have the SAME NUMBER of sentences as the input (±1).
- The output must be SHORTER than or equal to the input in word count.
- NEVER add information, examples, explanations, lists, or conclusions.
- NEVER split one sentence into a paragraph or numbered list.
- If the speaker said 2 things, output 2 things. Nothing more."""

POLISH_USER_TEMPLATE = """\
Clean up this raw transcription. Keep the same number of sentences. Remove filler words, fix grammar, make it read smoothly. Do NOT add any new content.

Raw transcription:
{text}"""

POLISH_USER_TEMPLATE_STYLED = """\
Clean up this raw transcription. Keep the same number of sentences. Remove filler words, fix grammar, make it read smoothly. Do NOT add any new content.

While cleaning, use this person's writing voice — their word choices and sentence patterns — but do NOT add new ideas or extra sentences:
{style}

Raw transcription:
{text}"""


MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def polish_stream(raw_text: str):
    """Yield polished tokens from Ollama as they arrive.

    Yields dicts: ``{"token": str}`` for content, ``{"error": str}`` on
    failure, and ``{"done": True}`` when finished.
    Retries up to MAX_RETRIES times on 500 errors (usually GPU memory pressure).
    """
    style = load_style()
    if style:
        user_msg = POLISH_USER_TEMPLATE_STYLED.format(style=style, text=raw_text)
    else:
        user_msg = POLISH_USER_TEMPLATE.format(text=raw_text)

    token_count = 0
    t_start = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("Sending polish request to Ollama (%s) – %d chars of input (attempt %d/%d)",
                     OLLAMA_MODEL, len(raw_text), attempt, MAX_RETRIES)
            t_req = time.perf_counter()
            resp = requests.post(
                OLLAMA_CHAT_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": POLISH_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
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
                    token = chunk.get("message", {}).get("content", "")
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
            log.error("Cannot connect to Ollama at %s", OLLAMA_CHAT_URL)
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
