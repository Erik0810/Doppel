"""
Writing-style extractor – analyses uploaded documents via Ollama/Mistral
and produces a bullet-point style profile saved to disk.
"""

import json
import logging
import os
import time

import requests

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

STYLE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "writing_style.json")

STYLE_EXTRACT_PROMPT = """\
You extract mechanical writing style patterns from text samples. Your output will be used as instructions to make OTHER text match this writer's style. The instructions must work for ANY topic.

EXAMPLES OF GOOD BULLETS (specific, actionable, topic-free):
- "Sentences average 20-30 words, mixing long compound sentences with occasional short punchy ones"
- "Heavily uses passive voice ('is demonstrated', 'was observed', 'can be seen')"
- "Opens paragraphs with a topic sentence, then expands with 2-3 supporting sentences"
- "Frequently uses hedging: 'may', 'could', 'suggests', 'it is possible that'"
- "Connects ideas with transitions like 'furthermore', 'however', 'in contrast', 'this highlights'"
- "Uses parenthetical asides to add clarification mid-sentence"
- "Writes in third person impersonal — avoids 'I' and 'you'"
- "Tends to restate a point in different words immediately after making it"
- "Uses colons to introduce explanations or elaborations"

EXAMPLES OF BAD BULLETS (too vague, or mentions content):
- "Use of academic references" ← BAD, refers to content
- "Informative and educational tone" ← BAD, too vague
- "Good use of grammar" ← BAD, meaningless
- "Discusses social implications" ← BAD, refers to content
- "Uses technical terms relevant to the topic" ← BAD, refers to content

Produce 8-15 bullets like the GOOD examples above. Every bullet must be a concrete, actionable writing instruction. Output ONLY the bullet list — no preamble, no headers."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    from PyPDF2 import PdfReader
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    from docx import Document
    import io
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def learn_style(combined_text: str):
    """Send combined document text to Ollama and yield progress/result dicts.

    Yields:
        {"status": "learning"}           – work has started
        {"token": str}                   – streamed token from the model
        {"done": True, "style": str}     – final style string
        {"error": str}                   – on failure
    """
    # Truncate to ~12k chars to stay within context window
    max_chars = 12000
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars]
        log.info("Truncated input text to %d chars for style extraction", max_chars)

    yield {"status": "learning"}

    try:
        log.info("Sending style-extraction request to Ollama (%s) – %d chars",
                 OLLAMA_MODEL, len(combined_text))

        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": (
                    "Analyse ONLY the writing style of the following text. "
                    "Do NOT summarize, rephrase, or respond to the content.\n\n"
                    "--- BEGIN WRITING SAMPLE ---\n"
                    f"{combined_text}\n"
                    "--- END WRITING SAMPLE ---\n\n"
                    "Now list the stylistic patterns as bullet points:"
                ),
                "system": STYLE_EXTRACT_PROMPT,
                "stream": True,
            },
            stream=True,
            timeout=180,
        )
        resp.raise_for_status()

        style_parts = []
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    style_parts.append(token)
                    yield {"token": token}
                if chunk.get("done"):
                    break

        full_style = "".join(style_parts).strip()
        save_style(full_style)
        yield {"done": True, "style": full_style}

    except requests.ConnectionError:
        log.error("Cannot connect to Ollama at %s", OLLAMA_URL)
        yield {"error": "Cannot connect to Ollama. Is it running? (ollama serve)"}
    except Exception as e:
        log.exception("Unexpected error during style extraction")
        yield {"error": str(e)}


def save_style(style_text: str):
    data = {"style": style_text}
    with open(STYLE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("Writing style saved to %s", STYLE_FILE)


def load_style() -> str | None:
    if os.path.exists(STYLE_FILE):
        with open(STYLE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("style")
    return None
