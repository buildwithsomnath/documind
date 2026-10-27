
import os

import psycopg
from dotenv import load_dotenv
from flask import Flask, jsonify

load_dotenv()

app = Flask(__name__)

DATABASE_URL = os.environ["DATABASE_URL"]


@app.get("/api/health")
def health():
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()

        return jsonify({
            "status": "ok",
            "database": "connected",
            "service": "DocuMind"
        }), 200

    except Exception:
        app.logger.exception("Health check failed")
        return jsonify({
            "status": "error",
            "database": "unavailable"
        }), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
