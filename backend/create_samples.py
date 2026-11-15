import io
from pathlib import Path
import pymupdf
import docx

sample_dir = Path(__file__).resolve().parent.parent / "sample_docs"
sample_dir.mkdir(parents=True, exist_ok=True)

# 1. Create sample PDF
pdf_path = sample_dir / "sample_architecture.pdf"
doc = pymupdf.open()

# Page 1
page1 = doc.new_page()
p1_text = (
    "DocuMind System Architecture\n\n"
    "DocuMind is a production-grade Retrieval-Augmented Generation (RAG) platform. "
    "It processes enterprise PDF and DOCX documents, decomposes them into semantically coherent "
    "chunks, generates dense vector representations, and provides high-precision question answering.\n\n"
    "Key architectural layers include the document ingestion pipeline, pgvector relational storage, "
    "hybrid reciprocal rank fusion (RRF) search, and cross-encoder reranking."
)
page1.insert_text((50, 72), p1_text, fontsize=12)

# Page 2
page2 = doc.new_page()
p2_text = (
    "Storage and Vector Indexing\n\n"
    "DocuMind utilizes PostgreSQL 16 equipped with the pgvector extension. "
    "Embeddings are computed using sentence-transformers/all-MiniLM-L6-v2, producing 384-dimensional dense vectors.\n\n"
    "To accelerate approximate nearest neighbor (ANN) vector queries, an HNSW index with cosine distance "
    "(vector_cosine_ops) is constructed over the document_chunks table. This achieves sub-millisecond retrieval "
    "while maintaining high recall."
)
page2.insert_text((50, 72), p2_text, fontsize=12)

doc.save(str(pdf_path))
doc.close()
print(f"Created sample PDF at: {pdf_path}")

# 2. Create sample DOCX
docx_path = sample_dir / "sample_spec.docx"
d = docx.Document()
d.add_heading("DocuMind Ingestion Specification", level=1)
d.add_paragraph(
    "This specification defines the ingestion requirements for text extraction and chunking. "
    "All incoming files must be validated for allowed MIME types and file extensions (PDF and DOCX). "
    "A maximum upload threshold of 16 MB is enforced."
)
d.add_heading("Page-Aware Text Chunking", level=2)
d.add_paragraph(
    "To preserve source provenance, each chunk retains the starting page, ending page, and paragraph offsets. "
    "A chunk size of 500 characters paired with a 100-character sliding overlap prevents context fragmentation "
    "while adhering to the 256-token limit of the all-MiniLM-L6-v2 embedding model."
)
d.save(str(docx_path))
print(f"Created sample DOCX at: {docx_path}")
