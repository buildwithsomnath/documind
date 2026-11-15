import math
import sys
from pathlib import Path
import pytest

# Ensure backend and root directories are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    calculate_faithfulness,
)
from evaluation.evaluator import RAGEvaluator


def test_recall_at_k():
    ground_truth = {"chunk-A", "chunk-B"}
    retrieved = ["chunk-A", "chunk-C", "chunk-B", "chunk-D"]

    # At k=1: only chunk-A retrieved -> Recall = 1/2 = 0.5
    assert recall_at_k(retrieved, ground_truth, k=1) == 0.5
    # At k=2: chunk-A and chunk-C -> Recall = 1/2 = 0.5
    assert recall_at_k(retrieved, ground_truth, k=2) == 0.5
    # At k=3: chunk-A, C, B -> Recall = 2/2 = 1.0
    assert recall_at_k(retrieved, ground_truth, k=3) == 1.0


def test_reciprocal_rank():
    ground_truth = {"target-1"}
    # Hit at rank 1 -> 1/1 = 1.0
    assert reciprocal_rank(["target-1", "other"], ground_truth) == 1.0
    # Hit at rank 2 -> 1/2 = 0.5
    assert reciprocal_rank(["other", "target-1"], ground_truth) == 0.5
    # Hit at rank 4 -> 1/4 = 0.25
    assert reciprocal_rank(["o1", "o2", "o3", "target-1"], ground_truth) == 0.25
    # Miss -> 0.0
    assert reciprocal_rank(["o1", "o2"], ground_truth) == 0.0


def test_mean_reciprocal_rank():
    queries_ret = [
        ["hit", "miss"],          # RR = 1.0
        ["miss", "hit"],          # RR = 0.5
        ["miss", "miss", "hit"],  # RR = 0.3333
    ]
    queries_gt = [{"hit"}, {"hit"}, {"hit"}]

    mrr = mean_reciprocal_rank(queries_ret, queries_gt)
    expected = (1.0 + 0.5 + 0.3333) / 3.0
    assert abs(mrr - expected) < 0.01


def test_ndcg_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    # Ideal ranking: doc1 is highly relevant (2), doc2 is relevant (1), doc3 is not relevant (0)
    relevance = {"doc1": 2.0, "doc2": 1.0, "doc3": 0.0}

    # Perfect ranking should have nDCG = 1.0
    assert ndcg_at_k(retrieved, relevance, k=3) == 1.0

    # Sub-optimal ranking: doc3 (0) before doc1 (2)
    bad_retrieved = ["doc3", "doc2", "doc1"]
    bad_ndcg = ndcg_at_k(bad_retrieved, relevance, k=3)
    assert bad_ndcg < 1.0
    assert bad_ndcg > 0.0


def test_faithfulness_calculation():
    context = [
        "DocuMind utilizes PostgreSQL 16 equipped with pgvector extension.",
        "Embeddings are computed using sentence-transformers/all-MiniLM-L6-v2."
    ]

    # Factual grounded answer
    grounded_answer = "PostgreSQL 16 with pgvector stores embeddings computed using all-MiniLM-L6-v2."
    result = calculate_faithfulness(grounded_answer, context)
    assert result["faithfulness_score"] == 1.0

    # Hallucinated answer
    hallucinated_answer = "MongoDB Atlas cluster stores Pinecone vector embeddings with OpenAI Ada model."
    bad_result = calculate_faithfulness(hallucinated_answer, context)
    assert bad_result["faithfulness_score"] == 0.0


def test_rag_evaluator_dataset_loading():
    evaluator = RAGEvaluator()
    assert len(evaluator.dataset) >= 4
    queries = [item["id"] for item in evaluator.dataset]
    assert "q1_vector_indexing" in queries
    assert "q4_negative_hallucination" in queries
