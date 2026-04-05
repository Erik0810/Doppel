"""
Reusable Whisper v3 Large Turbo transcription component.

Captures audio from the microphone in chunks and transcribes using
openai/whisper-large-v3-turbo via Hugging Face Transformers.
"""

import logging
import threading
import time
import warnings
import gc
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Silence the harmless duplicate-logits-processor and sequential-pipeline warnings
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono audio
SLICE_DURATION = 0.5  # seconds per mic read (small for low latency)
TRANSCRIBE_INTERVAL = 5.0  # transcribe accumulated audio every N seconds
OVERLAP_DURATION = 0.5  # seconds of trailing audio kept across chunks
SILENCE_THRESHOLD = 0.02  # ignore chunks quieter than this


class WhisperTranscriber:
    """Streams microphone audio and transcribes it with Whisper large-v3-turbo."""

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo"):
        self.model_id = model_id
        self._lock = threading.Lock()
        self._audio_buffer: list[np.ndarray] = []
        self._recording = False
        self._record_thread: threading.Thread | None = None
        self._transcribe_thread: threading.Thread | None = None
        self._on_text = None  # callback(text: str)
        self._pipe = None
        self._model = None  # keep a reference for unloading
        self._overlap: np.ndarray | None = None  # tail audio carried across chunks
        self._prev_words: list[str] = []  # trailing words from last chunk for dedup

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        """Load the Whisper model and build the HF pipeline."""
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self._torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self._device)
        self._model = model

        self._processor = AutoProcessor.from_pretrained(self.model_id)

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=self._torch_dtype,
            device=self._device,
        )

    def offload_to_cpu(self):
        """Move Whisper to CPU and free GPU VRAM so Ollama can use it."""
        if self._model is not None and self._device != "cpu":
            self._model.to("cpu")
            # Pipeline caches the device, so destroy and let
            # _move_to_gpu rebuild it
            self._pipe = None
            gc.collect()
            torch.cuda.empty_cache()

    def _move_to_gpu(self):
        """Move Whisper back to GPU and rebuild the pipeline (fast, no disk I/O)."""
        if self._model is None:
            self.load_model()
            return
        self._model.to(self._device)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self._model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            torch_dtype=self._torch_dtype,
            device=self._device,
        )

    def unload_model(self):
        """Move Whisper off GPU and free VRAM so Ollama can use it."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def _record_loop(self):
        """Continuously read small slices from the mic into a shared buffer."""
        import sounddevice as sd

        slice_samples = int(SAMPLE_RATE * SLICE_DURATION)

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while self._recording:
                frames, _ = stream.read(slice_samples)
                with self._lock:
                    self._audio_buffer.append(frames[:, 0].copy())

    def _transcribe_loop(self):
        """Periodically grab whatever audio has accumulated and transcribe it."""
        while self._recording:
            time.sleep(TRANSCRIBE_INTERVAL)

            with self._lock:
                if not self._audio_buffer:
                    continue
                chunk = np.concatenate(self._audio_buffer)
                self._audio_buffer.clear()

            # Prepend overlap from previous chunk to avoid cutting words
            if self._overlap is not None:
                chunk = np.concatenate([self._overlap, chunk])

            # Save tail as overlap for next iteration
            overlap_samples = int(SAMPLE_RATE * OVERLAP_DURATION)
            self._overlap = chunk[-overlap_samples:] if len(chunk) > overlap_samples else chunk.copy()

            # Skip silence / very quiet chunks
            if np.max(np.abs(chunk)) < SILENCE_THRESHOLD:
                continue

            result = self._pipe(
                {"raw": chunk, "sampling_rate": SAMPLE_RATE},
                generate_kwargs={"language": "en", "num_beams": 5},
                return_timestamps=True,
            )
            text = result["text"].strip()
            if text and self._on_text:
                text = self._dedup(text)
                if text:
                    self._on_text(text)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def _dedup(self, text: str) -> str:
        """Strip leading words that duplicate the tail of the previous chunk."""
        words = text.split()
        if self._prev_words and words:
            # Find the longest suffix of prev_words that matches a prefix of words
            max_check = min(len(self._prev_words), len(words))
            overlap_len = 0
            for n in range(1, max_check + 1):
                if [w.lower() for w in self._prev_words[-n:]] == [w.lower() for w in words[:n]]:
                    overlap_len = n
            words = words[overlap_len:]
        # Keep trailing words for next call (enough to cover the overlap window)
        self._prev_words = text.split()[-8:]
        return " ".join(words)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def start(self, on_text=None):
        """Begin recording and transcribing. *on_text(str)* is called for every chunk."""
        if self._pipe is None:
            self.load_model()

        self._on_text = on_text
        self._recording = True

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)
        self._record_thread.start()
        self._transcribe_thread.start()

    def stop(self):
        """Stop recording and wait for threads to finish."""
        self._recording = False
        if self._record_thread:
            self._record_thread.join()
        if self._transcribe_thread:
            self._transcribe_thread.join()
        with self._lock:
            self._audio_buffer.clear()
        self._overlap = None
        self._prev_words = []

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file and return the text."""
        if self._pipe is None:
            self.load_model()
        return self._pipe(path)["text"].strip()
