import logging
import sys
from pathlib import Path
from flask import Blueprint, request, jsonify

# Ensure documind root directory is in sys.path
# Level 1: routes, Level 2: app, Level 3: backend, Level 4: documind root
documind_root = Path(__file__).resolve().parent.parent.parent.parent
if str(documind_root) not in sys.path:
    sys.path.insert(0, str(documind_root))

logger = logging.getLogger(__name__)

evaluation_bp = Blueprint("evaluation", __name__, url_prefix="/api/evaluation")


@evaluation_bp.post("/run")
def run_benchmark():
    """
    Executes the full RAG benchmark:
    - Compares Vector Search, Keyword Search, and Hybrid Reranked on Recall@K, MRR, nDCG@3
    - Evaluates Citation Accuracy, Faithfulness, and Anti-Hallucination Guardrails
    """
    from evaluation.evaluator import RAGEvaluator
    data = request.get_json(silent=True) or {}
    skip_llm = data.get("skip_llm", False)

    try:
        evaluator = RAGEvaluator()
        report = evaluator.run_full_benchmark(run_generation=not skip_llm)
        return jsonify({
            "status": "success",
            "benchmark_results": report
        }), 200

    except Exception as e:
        logger.exception("Error executing evaluation benchmark")
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Benchmark evaluation failed: {str(e)}"
        }), 500
