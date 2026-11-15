import io
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import pytest
import pymupdf
import docx

from app.ingestion.extractor import DocumentExtractor, clean_text
from app.ingestion.chunker import PageAwareChunker
from app.ingestion.embedder import EmbeddingService
from app import app


@pytest.fixture
def sample_pdf_bytes():
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "DocuMind Architecture Overview. This platform processes documents with AI. High precision retrieval is supported.")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page two contains implementation details. Postgres and pgvector store vector embeddings.")
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_docx_bytes():
    d = docx.Document()
    d.add_paragraph("DocuMind Technical Specification. Production RAG pipeline with hybrid search.")
    d.add_paragraph("Section 2: Embedding Generation. MiniLM produces 384-dimensional vectors.")
    stream = io.BytesIO()
    d.save(stream)
    return stream.getvalue()


def test_clean_text():
    dirty = "Hello   world.\r\nThis is a test-\nword. \n\n\n\nExtra lines."
    cleaned = clean_text(dirty)
    assert "testword" in cleaned
    assert "   " not in cleaned
    assert "\r" not in cleaned


def test_pdf_extraction(sample_pdf_bytes):
    extracted = DocumentExtractor.extract_pdf(sample_pdf_bytes, "test.pdf")
    assert extracted.file_type == "pdf"
    assert extracted.total_pages == 2
    assert len(extracted.pages) == 2
    assert "DocuMind Architecture" in extracted.pages[0].text
    assert "Page two" in extracted.pages[1].text


def test_docx_extraction(sample_docx_bytes):
    extracted = DocumentExtractor.extract_docx(sample_docx_bytes, "test.docx")
    assert extracted.file_type == "docx"
    assert extracted.total_pages >= 1
    assert "DocuMind Technical Specification" in extracted.pages[0].text


def test_page_aware_chunking(sample_pdf_bytes):
    extracted = DocumentExtractor.extract_pdf(sample_pdf_bytes, "test.pdf")
    chunker = PageAwareChunker(chunk_size=60, chunk_overlap=15)
    chunks = chunker.chunk_pages(extracted.pages)

    assert len(chunks) > 0
    # Every chunk must have page numbers and sequential indices
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.page_number in (1, 2)
        assert len(c.content) > 0
        assert "page_number" in c.metadata


def test_embedding_service():
    service = EmbeddingService.get_instance()
    # Mock model for offline unit testing
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.05] * 384, [0.08] * 384]
    mock_model.get_sentence_embedding_dimension.return_value = 384
    service._model = mock_model
    service.dimension = 384

    texts = ["DocuMind retrieval test", "PostgreSQL pgvector storage"]
    embeddings = service.embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384


def test_upload_missing_file():
    client = app.test_client()
    resp = client.post("/api/documents/upload")
    assert resp.status_code == 400
    assert "Missing 'file' field" in resp.get_json()["message"]


def test_upload_invalid_extension():
    client = app.test_client()
    data = {"file": (io.BytesIO(b"fake image content"), "photo.png")}
    resp = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "not supported" in resp.get_json()["message"]
