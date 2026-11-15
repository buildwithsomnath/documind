import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import app
from app.retrieval.types import SearchResult
from app.retrieval.vector_search import VectorSearchEngine
from app.retrieval.keyword_search import KeywordSearchEngine
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import RerankerService, sigmoid


def test_search_result_to_dict():
    res = SearchResult(
        chunk_id="test-chunk-id",
        document_id="test-doc-id",
        filename="test.pdf",
        chunk_index=0,
        page_number=1,
        page_end=1,
        content="This is sample content",
        score=0.891234,
        retrieval_method="vector",
        metadata={"char_count": 22}
    )
    d = res.to_dict()
    assert d["chunk_id"] == "test-chunk-id"
    assert d["score"] == 0.8912
    assert d["retrieval_method"] == "vector"
    assert d["page_number"] == 1


def test_vector_search_empty_query():
    engine = VectorSearchEngine()
    results = engine.search(query="")
    assert results == []
    results = engine.search(query="   ")
    assert results == []


def test_keyword_search_empty_query():
    engine = KeywordSearchEngine()
    results = engine.search(query="")
    assert results == []


def test_vector_search_endpoint_missing_query():
    client = app.test_client()
    resp = client.post("/api/search/vector", json={})
    assert resp.status_code == 400
    assert "query" in resp.get_json()["message"]


def test_vector_search_endpoint_invalid_uuid():
    client = app.test_client()
    resp = client.post("/api/search/vector", json={"query": "test", "document_id": "not-a-uuid"})
    assert resp.status_code == 400
    assert "Invalid document_id" in resp.get_json()["message"]


def test_keyword_search_endpoint_missing_query():
    client = app.test_client()
    resp = client.post("/api/search/keyword", json={})
    assert resp.status_code == 400
    assert "query" in resp.get_json()["message"]


def test_sigmoid():
    assert abs(sigmoid(0.0) - 0.5) < 1e-4
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01


def test_reciprocal_rank_fusion_math():
    # Construct 2 mock candidate lists
    vec_chunk1 = SearchResult(
        chunk_id="c1", document_id="d1", filename="f1.pdf", chunk_index=0,
        page_number=1, page_end=1, content="Passage 1", score=0.9, retrieval_method="vector"
    )
    vec_chunk2 = SearchResult(
        chunk_id="c2", document_id="d1", filename="f1.pdf", chunk_index=1,
        page_number=2, page_end=2, content="Passage 2", score=0.8, retrieval_method="vector"
    )

    kw_chunk1 = SearchResult(
        chunk_id="c1", document_id="d1", filename="f1.pdf", chunk_index=0,
        page_number=1, page_end=1, content="Passage 1", score=0.5, retrieval_method="keyword"
    )
    kw_chunk3 = SearchResult(
        chunk_id="c3", document_id="d1", filename="f1.pdf", chunk_index=2,
        page_number=3, page_end=3, content="Passage 3", score=0.4, retrieval_method="keyword"
    )

    # c1 is rank 1 in both: score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032786
    # c2 is rank 2 in vec only: score = 1/(60+2) = 1/62 ≈ 0.016129
    # c3 is rank 2 in kw only: score = 1/(60+2) = 1/62 ≈ 0.016129
    fused = reciprocal_rank_fusion(
        vector_results=[vec_chunk1, vec_chunk2],
        keyword_results=[kw_chunk1, kw_chunk3],
        rrf_k=60,
        top_n=5
    )

    assert len(fused) == 3
    # c1 MUST be first due to consensus boost
    assert fused[0].chunk_id == "c1"
    assert abs(fused[0].score - (2.0 / 61.0)) < 1e-5
    assert fused[0].metadata["rrf_provenance"]["found_in_both"] is True

    # c2 and c3 follow
    assert fused[1].chunk_id in ("c2", "c3")
    assert abs(fused[1].score - (1.0 / 62.0)) < 1e-5


def test_reranker_service_mocked():
    service = RerankerService.get_instance()
    mock_model = MagicMock()
    # Mock logits: second pair is much more relevant
    mock_model.predict.return_value = [1.2, 5.0]
    service._model = mock_model

    candidates = [
        SearchResult("c1", "d1", "f.pdf", 0, 1, 1, "Moderate match", 0.5, "hybrid_rrf"),
        SearchResult("c2", "d1", "f.pdf", 1, 1, 1, "Exact target match", 0.3, "hybrid_rrf"),
    ]

    reranked = service.rerank("target query", candidates, top_k=2)
    assert len(reranked) == 2
    # c2 had higher logit (5.0 vs 1.2), so it should be reranked to index 0!
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score > reranked[1].score
    assert reranked[0].retrieval_method == "hybrid_reranked"


def test_hybrid_search_endpoint_missing_query():
    client = app.test_client()
    resp = client.post("/api/search/hybrid", json={})
    assert resp.status_code == 400
    assert "query" in resp.get_json()["message"]


def test_hybrid_search_endpoint_invalid_uuid():
    client = app.test_client()
    resp = client.post("/api/search/hybrid", json={"query": "test", "document_id": "not-valid"})
    assert resp.status_code == 400
    assert "Invalid document_id" in resp.get_json()["message"]
