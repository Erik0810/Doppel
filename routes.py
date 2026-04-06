"""
Flask routes – thin web layer over DoppelEngine.
"""

import json
import queue

from flask import Blueprint, render_template, Response, request, jsonify

from core.engine import DoppelEngine

bp = Blueprint("main", __name__)

# The engine instance is set by app.py at startup
engine: DoppelEngine | None = None


def init_engine(eng: DoppelEngine):
    global engine
    engine = eng


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/start", methods=["POST"])
def start_recording():
    engine.start_recording()
    return jsonify(ok=True)


@bp.route("/stop", methods=["POST"])
def stop_recording():
    engine.stop_recording()
    return jsonify(ok=True)


@bp.route("/stream")
def stream():
    """Server-Sent Events endpoint that pushes transcribed text chunks."""
    def generate():
        while True:
            try:
                text = engine.text_queue.get(timeout=30)
                if text is None:
                    # Sentinel received – None means the transcriber
                    # is done and no more text will arrive. Forward
                    # a done event to the browser so it can move on.
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
                yield f"data: {json.dumps({'text': text})}\n\n"
            except queue.Empty:
                yield ": keepalive\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@bp.route("/polish", methods=["POST"])
def polish():
    """Stream-polish raw transcription text via Ollama."""
    raw_text = request.json.get("text", "").strip()
    if not raw_text:
        return jsonify(error="No text provided"), 400

    def generate():
        for chunk in engine.polish(raw_text):
            yield f"data: {json.dumps(chunk)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


ALLOWED_EXTENSIONS = {".pdf", ".docx"}


@bp.route("/learn-style", methods=["POST"])
def learn_style():
    """Accept file uploads, extract text, and stream style-learning progress."""
    import os

    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files uploaded"), 400

    # Extract text from all uploaded files
    combined_parts = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify(error=f"Unsupported file type: {f.filename}"), 400
        try:
            file_bytes = f.read()
            text = engine.extract_file_text(f.filename, file_bytes)
            if text.strip():
                combined_parts.append(text)
        except Exception as e:
            return jsonify(error=f"Failed to read {f.filename}: {e}"), 400

    if not combined_parts:
        return jsonify(error="No text could be extracted from the uploaded files"), 400

    combined_text = "\n\n---\n\n".join(combined_parts)

    def generate():
        for chunk in engine.learn_writing_style(combined_text):
            yield f"data: {json.dumps(chunk)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@bp.route("/writing-style")
def writing_style():
    """Return the currently saved writing style."""
    style = engine.get_writing_style()
    return jsonify(style=style)
