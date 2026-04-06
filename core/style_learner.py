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
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"
STYLE_LEARN_MODEL = "deepseek-r1:14b"

STYLE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "writing_style.json")

STYLE_SYSTEM_MSG = "You are a linguistic analyst. You ONLY analyse writing mechanics. You ALWAYS respond in English regardless of the input language."

STYLE_PASS1_TEMPLATE = """\
I will show you a text. Do NOT read it for meaning. Instead, answer these specific questions about HOW it is written. Respond in English.

TEXT:
\"\"\"
{text}
\"\"\"

Answer each question with specific evidence. Do NOT discuss the topic or content.

1. What is the average sentence length in words? Count 5 representative sentences and give their word counts.
2. List 5 exact sentence-opening words or phrases from the text (the first 3-4 words of different sentences).
3. Find 3 verb phrases. Are they active ("X causes Y") or passive ("Y is caused by X")? Quote them.
4. List every transition word or connecting phrase you can find (e.g. "however", "this means", "furthermore", "in addition").
5. Find 3 examples of either hedging ("may", "could", "suggests") or certainty ("clearly", "demonstrates", "shows"). Quote them exactly.
6. Pick 2 sentences and describe their clause structure — are they simple (one clause) or complex (main + subordinate clauses)?
7. List 5 vocabulary words that indicate the formality level. Are they everyday words or elevated/formal ones?
8. Look at any 3 consecutive sentences. How does sentence 2 connect to sentence 1? Does it repeat a keyword, use a pronoun reference, or use a transition word?"""

STYLE_PASS2_TEMPLATE = """\
Below is a mechanical analysis of someone's writing. Write a SHORT paragraph (3-5 sentences max) describing how this person writes. This paragraph will be given to another AI that cleans up speech transcriptions so they sound like this person wrote them.

ANALYSIS:
{analysis}

RULES FOR YOUR OUTPUT:
- Write a single short paragraph in plain prose, NOT a bullet list
- Describe the voice the way you'd describe a friend's writing: "They tend to write long formal sentences, favor passive voice like 'is demonstrated by', and hedge with words like 'may' and 'could'."
- Include specific example words and phrases in quotes
- Do NOT mention any topic, subject, or field
- Do NOT use imperative mood ("Use X", "Incorporate Y") — use descriptive mood ("They tend to X", "This writer favors Y")
- Respond in English
- Keep it under 100 words

Output ONLY the paragraph. No preamble, no headers."""


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


def _chat_collect(model: str, system: str, user: str, timeout: int = 600) -> str:
    """Send a chat request and collect the full response, stripping <think> blocks."""
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
        },
        stream=True,
        timeout=timeout,
    )
    resp.raise_for_status()

    tokens = []
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                tokens.append(token)
            if chunk.get("done"):
                break

    full = "".join(tokens).strip()
    # Strip thinking block if present
    if "</think>" in full:
        full = full.split("</think>", 1)[1].strip()
    return full


def learn_style(combined_text: str):
    """Two-pass style extraction via deepseek-r1.

    Pass 1: Answer specific mechanical questions about the writing (forces
            the model to look at HOW the text is written).
    Pass 2: Convert the raw analysis into actionable style rules.

    Yields:
        {"status": str}                  – phase updates
        {"thinking": str}                – reasoning token (for UI feedback)
        {"token": str}                   – answer token from the model
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
        # ---- PASS 1: Mechanical analysis (non-streamed, collected) ----
        log.info("Pass 1: Extracting mechanical observations (%s) – %d chars",
                 STYLE_LEARN_MODEL, len(combined_text))
        yield {"status": "analyzing"}

        analysis = _chat_collect(
            STYLE_LEARN_MODEL,
            STYLE_SYSTEM_MSG,
            STYLE_PASS1_TEMPLATE.format(text=combined_text),
        )
        log.info("Pass 1 complete – %d chars of analysis", len(analysis))

        # ---- PASS 2: Synthesize into style rules (streamed) ----
        log.info("Pass 2: Synthesizing style rules")
        yield {"status": "synthesizing"}

        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": STYLE_LEARN_MODEL,
                "messages": [
                    {"role": "system", "content": STYLE_SYSTEM_MSG},
                    {"role": "user", "content": STYLE_PASS2_TEMPLATE.format(analysis=analysis)},
                ],
                "stream": True,
            },
            stream=True,
            timeout=600,
        )
        resp.raise_for_status()

        all_tokens = []
        in_thinking = False
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    all_tokens.append(token)
                    accumulated = "".join(all_tokens)

                    if "<think>" in accumulated and not in_thinking:
                        in_thinking = True
                    if in_thinking and "</think>" not in accumulated:
                        yield {"thinking": token}
                    elif in_thinking and "</think>" in accumulated:
                        in_thinking = False
                        yield {"thinking": token}
                    else:
                        yield {"token": token}

                if chunk.get("done"):
                    break

        full_output = "".join(all_tokens).strip()
        if "</think>" in full_output:
            full_style = full_output.split("</think>", 1)[1].strip()
        else:
            full_style = full_output

        save_style(full_style)
        yield {"done": True, "style": full_style}

    except requests.ConnectionError:
        log.error("Cannot connect to Ollama at %s", OLLAMA_URL)
        yield {"error": "Cannot connect to Ollama. Is it running? (ollama serve)"}
    except Exception as e:
        log.exception("Unexpected error during style extraction")
        yield {"error": str(e)}
    finally:
        # Unload deepseek-r1 from VRAM so it doesn't compete with
        # Whisper and Mistral during normal recording/polishing.
        _unload_model(STYLE_LEARN_MODEL)


def _unload_model(model_name: str):
    """Tell Ollama to immediately unload a model from memory."""
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": model_name, "keep_alive": 0},
            timeout=10,
        )
        log.info("Unloaded %s from Ollama VRAM", model_name)
    except Exception:
        log.warning("Failed to unload %s – it may linger in VRAM", model_name)


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
