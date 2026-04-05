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
        # Move Whisper back to GPU (fast – no disk I/O, just a memory copy)
        self.transcriber._move_to_gpu()
        self.transcriber.start(on_text=self._on_text)

    def stop_recording(self):
        self.transcriber.stop()
        # Move Whisper to CPU to free GPU VRAM for Ollama/Mistral polishing
        log.info("Moving Whisper to CPU to free VRAM for polishing")
        self.transcriber.offload_to_cpu()
        # Sentinel: push None into the queue to tell consumers
        # "no more text is coming." The SSE stream checks for this
        # and sends a done event to the browser.
        self.text_queue.put(None)

    # ------------------------------------------------------------------
    # Polishing
    # ------------------------------------------------------------------
    @staticmethod
    def polish(raw_text: str):
        """Yield dicts from the polisher stream (token / error / done)."""
        yield from polish_stream(raw_text)
