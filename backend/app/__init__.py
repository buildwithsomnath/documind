import os
import logging
import psycopg
from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config
from app.db import init_db
from app.routes.documents import documents_bp

logger = logging.getLogger(__name__)


def create_app(config_class=Config) -> Flask:
    """
    Application factory for DocuMind backend.
    Configures extensions, routes, and error handlers.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config_class.MAX_CONTENT_LENGTH

    # Enable CORS for frontend integration
    CORS(app)

    # Register blueprints
    app.register_blueprint(documents_bp)

    @app.get("/api/health")
    def health():
        """
        Health check endpoint verifying application status and PostgreSQL connectivity.
        Preserved exactly as verified in Phase 1, Step 1.
        """
        try:
            with psycopg.connect(config_class.DATABASE_URL) as conn:
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

    @app.errorhandler(413)
    def request_entity_too_large(error):
        max_mb = config_class.MAX_CONTENT_LENGTH // (1024 * 1024)
        return jsonify({
            "error": "Payload Too Large",
            "message": f"File exceeds maximum upload limit of {max_mb} MB."
        }), 413

    @app.cli.command("init-db")
    def init_db_command():
        """CLI command: flask init-db"""
        init_db()
        print("Database tables and pgvector extension initialized.")

    return app


# Default application instance for quick imports
app = create_app()
