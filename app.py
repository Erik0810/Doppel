"""
Doppel – entry point for the Flask web interface.
"""

import logging

from flask import Flask

from core.engine import DoppelEngine
from routes import bp, init_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

app = Flask(__name__)
engine = DoppelEngine()

init_engine(engine)
app.register_blueprint(bp)

if __name__ == "__main__":
    print("Loading Whisper model …")
    engine.load_model()
    print("Model loaded. Starting Flask server …")
    app.run(debug=False, threaded=True, port=5000)
