import logging
import uuid
from flask import Blueprint, request, jsonify

from app.generation.service import QAService

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")
qa_service = QAService()


@chat_bp.post("/ask")
def ask():
    """
    Endpoint for grounded Document Q&A using Google Gemini.

    Request JSON:
        - question (str, required): Natural language question
        - top_k (int, optional): Number of passages to supply to LLM (default: 3, max: 10)
        - document_id (str, optional): Target specific document UUID
        - enable_rerank (bool, optional): Enable Cross-Encoder reranking (default: true)
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question")

    if not question or not str(question).strip():
        return jsonify({
            "error": "Bad Request",
            "message": "Field 'question' is required and cannot be empty."
        }), 400

    top_k = max(1, min(data.get("top_k", 3), 10))
    enable_rerank = data.get("enable_rerank", True)
    if isinstance(enable_rerank, str):
        enable_rerank = enable_rerank.lower() == "true"

    doc_id_raw = data.get("document_id")
    document_id = None
    if doc_id_raw:
        try:
            document_id = uuid.UUID(doc_id_raw)
        except ValueError:
            return jsonify({
                "error": "Bad Request",
                "message": f"Invalid document_id format: '{doc_id_raw}'."
            }), 400

    try:
        response = qa_service.answer_question(
            question=question,
            top_k=top_k,
            document_id=document_id,
            enable_rerank=enable_rerank,
        )
        return jsonify(response), 200

    except ValueError as ve:
        # Configuration or missing API key issue
        logger.warning(f"Configuration or validation error in chat: {ve}")
        return jsonify({
            "error": "Configuration Error",
            "message": str(ve)
        }), 400

    except PermissionError as pe:
        logger.error(f"Authentication failure with Gemini API: {pe}")
        return jsonify({
            "error": "Authentication Error",
            "message": str(pe)
        }), 401

    except Exception as e:
        logger.exception("Unexpected error during Q&A processing")
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Failed to generate answer: {str(e)}"
        }), 500
