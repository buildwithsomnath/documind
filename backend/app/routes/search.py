import logging
import uuid
from flask import Blueprint, request, jsonify

from app.retrieval.vector_search import VectorSearchEngine
from app.retrieval.keyword_search import KeywordSearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)

search_bp = Blueprint("search", __name__, url_prefix="/api/search")

# Instantiate search engines
vector_engine = VectorSearchEngine()
keyword_engine = KeywordSearchEngine()
hybrid_engine = HybridSearchEngine()


@search_bp.post("/vector")
def search_vector():
    """
    Executes semantic vector search using pgvector cosine distance.

    Request JSON:
        - query (str, required): Natural language search query
        - top_k (int, optional): Number of results to return (default: 5)
        - document_id (str, optional): Constrain search to a specific document UUID
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query")

    if not query or not str(query).strip():
        return jsonify({
            "error": "Bad Request",
            "message": "Field 'query' is required and cannot be empty."
        }), 400

    top_k = max(1, min(data.get("top_k", 5), 50))

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
        results = vector_engine.search(query=query, top_k=top_k, document_id=document_id)
        return jsonify({
            "query": query,
            "retrieval_method": "vector",
            "total_results": len(results),
            "results": [r.to_dict() for r in results]
        }), 200

    except Exception as e:
        logger.exception("Error executing vector search")
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Vector search failed: {str(e)}"
        }), 500


@search_bp.post("/keyword")
def search_keyword():
    """
    Executes lexical keyword search using PostgreSQL Full-Text Search (FTS).

    Request JSON:
        - query (str, required): Search keywords / phrase
        - top_k (int, optional): Number of results to return (default: 5)
        - document_id (str, optional): Constrain search to a specific document UUID
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query")

    if not query or not str(query).strip():
        return jsonify({
            "error": "Bad Request",
            "message": "Field 'query' is required and cannot be empty."
        }), 400

    top_k = max(1, min(data.get("top_k", 5), 50))

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
        results = keyword_engine.search(query=query, top_k=top_k, document_id=document_id)
        return jsonify({
            "query": query,
            "retrieval_method": "keyword",
            "total_results": len(results),
            "results": [r.to_dict() for r in results]
        }), 200

    except Exception as e:
        logger.exception("Error executing keyword search")
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Keyword search failed: {str(e)}"
        }), 500


@search_bp.post("/hybrid")
def search_hybrid():
    """
    Executes two-stage Hybrid Retrieval:
    Vector Search + Full-Text Search -> Reciprocal Rank Fusion (RRF) -> Cross-Encoder Reranking.

    Request JSON:
        - query (str, required): Natural language search query
        - top_k (int, optional): Final number of reranked passages to return (default: 5)
        - document_id (str, optional): Constrain search to a specific document UUID
        - enable_rerank (bool, optional): Enable Cross-Encoder reranking (default: true)
        - candidate_k (int, optional): Candidate pool size from Stage 1 (default: 20)
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query")

    if not query or not str(query).strip():
        return jsonify({
            "error": "Bad Request",
            "message": "Field 'query' is required and cannot be empty."
        }), 400

    top_k = max(1, min(data.get("top_k", 5), 50))
    candidate_k = max(top_k, min(data.get("candidate_k", 20), 100))
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
        results = hybrid_engine.search(
            query=query,
            top_k=top_k,
            document_id=document_id,
            enable_rerank=enable_rerank,
            candidate_k=candidate_k,
        )

        method = "hybrid_reranked" if enable_rerank else "hybrid_rrf"
        return jsonify({
            "query": query,
            "retrieval_method": method,
            "enable_rerank": enable_rerank,
            "candidate_k": candidate_k,
            "total_results": len(results),
            "results": [r.to_dict() for r in results]
        }), 200

    except Exception as e:
        logger.exception("Error executing hybrid search")
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Hybrid search failed: {str(e)}"
        }), 500
