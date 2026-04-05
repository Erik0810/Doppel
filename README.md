# Doppel

Live speech-to-text transcription with automatic cleanup. Records from your mic, transcribes with Whisper, then runs the raw text through Ollama to clean up filler words and formatting.

Two-panel UI: left side shows raw transcription, right side shows the cleaned version.

![Usage example](static/usage_example.png)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- A mic

## Setup

```
pip install -r requirements.txt
ollama pull mistral
```

## Run

```
python app.py
```

Open http://localhost:5000 in your browser. Hit **Start Recording**, talk, hit **Stop**. Raw text appears on the left, cleaned text streams in on the right.

**Clear Page** wipes both panels. Recording again appends new text without clearing previous output.

## Roadmap

- [x] Smart GPU memory sharing between models
- [x] Backend/frontend separation
- [x] Auto-retry on model failures
- [ ] Desktop app (no browser needed)
- [ ] Packaged installer (.exe)
- [ ] Learn your writing style from uploaded docs
- [ ] UI overhaul
