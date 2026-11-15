import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import app
from app.retrieval.types import SearchResult
from app.generation.prompt import format_retrieved_context, build_grounded_prompt, SYSTEM_PROMPT
from app.generation.citations import CitationValidator
from app.generation.gemini_client import GeminiClient
from app.generation.service import QAService


@pytest.fixture
def mock_passages():
    return [
        SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="sample_spec.docx",
            chunk_index=0,
            page_number=1,
            page_end=1,
            content="A chunk size of 500 characters paired with 100-character overlap prevents fragmentation.",
            score=0.89,
            retrieval_method="hybrid_reranked"
        ),
        SearchResult(
            chunk_id="chunk-2",
            document_id="doc-2",
            filename="sample_architecture.pdf",
            chunk_index=1,
            page_number=2,
            page_end=2,
            content="PostgreSQL 16 with pgvector stores 384-dimensional dense vectors using HNSW indexing.",
            score=0.85,
            retrieval_method="hybrid_reranked"
        )
    ]


def test_format_retrieved_context(mock_passages):
    formatted, refs = format_retrieved_context(mock_passages)
    assert "[Passage 1] Source: sample_spec.docx | Page 1" in formatted
    assert "[Passage 2] Source: sample_architecture.pdf | Page 2" in formatted
    assert len(refs) == 2
    assert refs[0]["filename"] == "sample_spec.docx"
    assert refs[1]["filename"] == "sample_architecture.pdf"


def test_build_grounded_prompt(mock_passages):
    system_p, user_p, refs = build_grounded_prompt("What is the chunk size?", mock_passages)
    assert "MANDATORY GROUNDING RULES" in system_p
    assert "What is the chunk size?" in user_p
    assert "REFERENCE PASSAGES:" in user_p


def test_citation_validator_verified_and_unverified():
    retrieved = [
        {"filename": "sample_spec.docx", "page_number": 1, "chunk_id": "c1", "score": 0.9, "passage_index": 1},
        {"filename": "sample_architecture.pdf", "page_number": 2, "chunk_id": "c2", "score": 0.8, "passage_index": 2},
    ]

    answer_with_citations = (
        "The chunk size is 500 characters [sample_spec.docx, Page 1]. "
        "PostgreSQL pgvector stores vectors [sample_architecture.pdf, Page 2]. "
        "Also, some unrelated claim [unrelated_paper.pdf, Page 5]."
    )

    result = CitationValidator.extract_and_verify(answer_with_citations, retrieved)

    assert result["total_citations"] == 3
    assert result["verified_citations"] == 2
    assert abs(result["citation_accuracy_percentage"] - 66.7) < 0.5

    citations = {c["filename"]: c for c in result["citations"]}
    assert citations["sample_spec.docx"]["verified"] is True
    assert citations["sample_architecture.pdf"]["verified"] is True
    assert citations["unrelated_paper.pdf"]["verified"] is False


def test_gemini_client_unconfigured_error():
    client = GeminiClient(api_key="")
    assert client.is_configured() is False
    with pytest.raises(ValueError) as exc:
        client.generate("system", "prompt")
    assert "GEMINI_API_KEY is not configured" in str(exc.value)


def test_gemini_client_endpoint_url():
    client = GeminiClient(api_key="test-key", model_name="gemini-3.1-flash-lite")
    assert "models/gemini-3.1-flash-lite:generateContent" in client.endpoint_url


def test_qa_service_mocked_gemini(mock_passages):
    service = QAService()
    # Mock hybrid retrieval engine
    service.hybrid_engine = MagicMock()
    service.hybrid_engine.search.return_value = mock_passages

    # Mock Gemini client
    service.gemini_client = MagicMock()
    service.gemini_client.generate.return_value = (
        "The chunk size is 500 characters [sample_spec.docx, Page 1] and uses HNSW indexing [sample_architecture.pdf, Page 2]."
    )

    response = service.answer_question("What is the chunk size?")
    assert response["question"] == "What is the chunk size?"
    assert "500 characters" in response["answer"]
    assert len(response["citations"]) == 2
    assert response["citation_accuracy_percentage"] == 100.0
    assert "sample_spec.docx" in response["sources_used"]
    assert "sample_architecture.pdf" in response["sources_used"]


def test_chat_ask_missing_question():
    client = app.test_client()
    resp = client.post("/api/chat/ask", json={})
    assert resp.status_code == 400
    assert "question" in resp.get_json()["message"]


def test_chat_ask_invalid_document_uuid():
    client = app.test_client()
    resp = client.post("/api/chat/ask", json={"question": "Test question", "document_id": "not-a-uuid"})
    assert resp.status_code == 400
    assert "Invalid document_id" in resp.get_json()["message"]
