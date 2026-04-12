"""
Ollama-based text polisher – single-pass with deepseek-r1:14b.
"""

import json
import logging
import time
import requests

from core.style_learner import load_style

log = logging.getLogger(__name__)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "deepseek-r1:14b"

SYSTEM_PROMPT = """\
Clean up this speech transcription. Remove filler words, fix grammar, keep the same meaning and sentence count. Output ONLY the cleaned text."""

STYLED_SYSTEM_PROMPT = """\
Clean up this speech transcription. Remove filler words, fix grammar, keep the same meaning and sentence count. Lightly match this writing voice: {style}

Output ONLY the cleaned text."""


def unload_model():
    """Tell Ollama to unload deepseek from VRAM."""
    try:
        requests.post(
            OLLAMA_CHAT_URL,
            json={"model": MODEL, "keep_alive": 0},
            timeout=10,
        )
        log.info("Unloaded %s from Ollama VRAM", MODEL)
    except Exception:
        log.warning("Failed to unload %s from Ollama", MODEL)


def polish_stream(raw_text: str):
    """Single-pass polish with deepseek-r1:14b. Streams tokens to the client.

    Yields dicts: {"token": str}, {"error": str}, {"done": True}.
    """
    style = load_style()
    if style:
        system = STYLED_SYSTEM_PROMPT.format(style=style)
    else:
        system = SYSTEM_PROMPT

    input_words = len(raw_text.split())
    # deepseek-r1 uses ~200-500 tokens for <think> reasoning before output,
    # so we need a large budget: thinking overhead + actual output cap
    max_tokens = int(input_words * 1.5) + 600

    # Size the context window to what we actually need: system prompt
    # (~100 tokens) + input + generation budget, with a small margin.
    # This avoids allocating a large default KV-cache on low-VRAM GPUs.
    # ~1.4 tokens per English word is a conservative average for sub-word tokenizers.
    estimated_input_tokens = int(input_words * 1.4) + 200
    num_ctx = max(1024, estimated_input_tokens + max_tokens + 256)

    log.info("Polish: ~%d input words, cap %d tokens (incl. thinking), num_ctx=%d, style=%s",
             input_words, max_tokens, num_ctx, bool(style))

    t_start = time.perf_counter()
    token_count = 0

    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": raw_text},
                ],
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                    "num_ctx": num_ctx,
                    "num_batch": 128,
                },
            },
            stream=True,
            timeout=180,
        )
        resp.raise_for_status()

        in_think = False
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if chunk.get("done"):
                break
            if not token:
                continue
            if "<think>" in token:
                in_think = True
                continue
            if "</think>" in token:
                in_think = False
                continue
            if in_think:
                continue
            token_count += 1
            yield {"token": token}

    except requests.ConnectionError:
        yield {"error": "Cannot connect to Ollama. Is it running?"}
    except Exception as e:
        log.exception("Polish error")
        yield {"error": str(e)}

    unload_model()
    log.info("Polish done – %d tokens in %.1fs", token_count, time.perf_counter() - t_start)
    yield {"done": True}
