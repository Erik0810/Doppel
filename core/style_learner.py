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
You are a writing-style analyst. You do NOT summarize, rephrase, or respond to the content.
You ONLY analyse HOW the text is written — the style, not the substance.

Output a concise bullet-point list that captures the writer's unique style.

Focus on:
- Sentence structure and length preferences (short/punchy vs long/complex)
- Vocabulary level (casual, formal, academic, technical)
- Tone (e.g. conversational, authoritative, humorous, dry)
- Use of rhetorical devices (metaphors, analogies, repetition, etc.)
- Paragraph structure and transitions
- Point-of-view preferences (first person, third person, etc.)
- Any distinctive quirks or patterns

CRITICAL RULES:
- Do NOT summarize or rephrase the content of the text.
- Do NOT respond to the text as if it were a question or task.
- Do NOT mention specific topics, facts, or arguments from the text.
- ONLY describe stylistic patterns. Every bullet must be about HOW the author writes, not WHAT they wrote about.
- Output ONLY a bullet-point list. No preamble, no headers, no commentary.
- Each bullet should start with "- ".
- Aim for 8-15 bullet points."""


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
