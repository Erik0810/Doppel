"""
DoppelEngine – framework-agnostic orchestrator for transcription + polishing.

Any UI layer (Flask, desktop GUI, CLI) should create one DoppelEngine and
call its methods.  Text arrives via the *on_text* callback or the thread-safe
text_queue.
"""

import logging
import queue

from core.transcriber import WhisperTranscriber
from core.polisher import polish_stream
from core.style_learner import (
    extract_text_from_pdf,
    extract_text_from_docx,
    learn_style,
    load_style,
)

log = logging.getLogger(__name__)


class DoppelEngine:
    def __init__(self):
        self.transcriber = WhisperTranscriber()
        # Thread-safe queue so any frontend can consume transcribed chunks
        self.text_queue: queue.Queue[str] = queue.Queue()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load_model(self):
        """Pre-load the Whisper model (can take a while on first run)."""
        self.transcriber.load_model()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _on_text(self, text: str):
        self.text_queue.put(text)

    def start_recording(self):
        self.transcriber.start(on_text=self._on_text)

    def stop_recording(self):
        self.transcriber.stop()
        self.text_queue.put(None)

    # ------------------------------------------------------------------
    # Polishing
    # ------------------------------------------------------------------
    @staticmethod
    def polish(raw_text: str):
        """Yield dicts from the polisher stream (token / error / done)."""
        yield from polish_stream(raw_text)

    # ------------------------------------------------------------------
    # Writing-style learning
    # ------------------------------------------------------------------
    @staticmethod
    def extract_file_text(filename: str, file_bytes: bytes) -> str:
        """Extract plain text from an uploaded PDF or DOCX."""
        lower = filename.lower()
        if lower.endswith(".pdf"):
            return extract_text_from_pdf(file_bytes)
        elif lower.endswith(".docx"):
            return extract_text_from_docx(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    @staticmethod
    def learn_writing_style(combined_text: str):
        """Yield progress dicts from the style learner."""
        yield from learn_style(combined_text)

    @staticmethod
    def get_writing_style() -> str | None:
        """Return the saved writing style, or None."""
        return load_style()
