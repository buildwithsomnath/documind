import io
import logging
import uuid
from typing import Dict, Any, Union
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from app.config import Config
from app.db import get_db_session
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.ingestion.extractor import DocumentExtractor
from app.ingestion.chunker import PageAwareChunker
from app.ingestion.embedder import EmbeddingService

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Production-grade document ingestion pipeline:
    Extract -> Chunk -> Embed -> Persist in PostgreSQL (pgvector).
    """

    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ):
        self.chunker = PageAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingService.get_instance()

    @staticmethod
    def validate_file(file: FileStorage) -> str:
        """
        Validates the uploaded file object and returns the sanitized filename.
        Raises ValueError if validation fails.
        """
        if not file or not file.filename:
            raise ValueError("No file provided in the request.")

        filename = secure_filename(file.filename)
        if not filename:
            raise ValueError("Invalid filename.")

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in Config.ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(Config.ALLOWED_EXTENSIONS))
            raise ValueError(f"File type '.{ext}' is not supported. Allowed formats: {allowed}")

        return filename

    def process_file_storage(self, file: FileStorage) -> Dict[str, Any]:
        """
        Processes a file uploaded via multipart/form-data.
        """
        filename = self.validate_file(file)
        file_bytes = file.read()
        file_size = len(file_bytes)

        if file_size == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        if file_size > Config.MAX_CONTENT_LENGTH:
            raise ValueError(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum "
                f"allowed limit of {Config.MAX_CONTENT_LENGTH / (1024 * 1024):.0f} MB."
            )

        ext = filename.rsplit(".", 1)[-1].lower()
        return self._run_pipeline(file_bytes=file_bytes, filename=filename, file_type=ext, file_size=file_size)

    def _run_pipeline(self, file_bytes: bytes, filename: str, file_type: str, file_size: int) -> Dict[str, Any]:
        """
        Executes the extraction, chunking, embedding, and storage pipeline.
        Ensures atomicity: if embedding or database persistence fails,
        the transaction is rolled back or marked as failed with the error recorded.
        """
        doc_id = uuid.uuid4()
        logger.info(f"Starting ingestion for '{filename}' ({file_size} bytes, id={doc_id})")

        with get_db_session() as session:
            # 1. Create document record in 'processing' status
            document = Document(
                id=doc_id,
                filename=filename,
                file_type=file_type,
                file_size_bytes=file_size,
                status="processing",
            )
            session.add(document)
            session.flush()

            try:
                # 2. Extract text & page metadata
                logger.info(f"Extracting text from {filename}...")
                extracted_doc = DocumentExtractor.extract(file_source=file_bytes, filename=filename)
                document.total_pages = extracted_doc.total_pages

                if not extracted_doc.pages or extracted_doc.total_words == 0:
                    raise ValueError(
                        f"Document '{filename}' contains no readable text. "
                        "If this is a scanned PDF, optical character recognition (OCR) is required."
                    )

                # 3. Page-aware overlapping chunking
                logger.info(f"Chunking {len(extracted_doc.pages)} pages...")
                chunks = self.chunker.chunk_pages(extracted_doc.pages)
                if not chunks:
                    raise ValueError("Chunking produced 0 chunks from the extracted text.")

                # 4. Generate vector embeddings in batch
                logger.info(f"Generating embeddings for {len(chunks)} chunk(s)...")
                chunk_texts = [c.content for c in chunks]
                embeddings = self.embedder.embed_texts(chunk_texts, batch_size=32)

                # 5. Persist chunks with embeddings to PostgreSQL
                logger.info(f"Storing {len(chunks)} chunks in PostgreSQL...")
                chunk_models = []
                for chunk, embedding in zip(chunks, embeddings):
                    chunk_model = DocumentChunk(
                        document_id=doc_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        page_number=chunk.page_number,
                        page_end=chunk.page_end,
                        chunk_metadata=chunk.metadata,
                        embedding=embedding,
                    )
                    chunk_models.append(chunk_model)

                session.add_all(chunk_models)

                # 6. Update document status to completed
                document.total_chunks = len(chunks)
                document.status = "completed"
                document.error_message = None

                logger.info(
                    f"Successfully ingested '{filename}': {document.total_pages} page(s), "
                    f"{document.total_chunks} chunk(s)."
                )

                # Return serializable dict before session closes
                doc_dict = document.to_dict()
                doc_dict["chunks_preview"] = [
                    {
                        "chunk_index": c.chunk_index,
                        "page_number": c.page_number,
                        "content_preview": c.content[:120] + "..." if len(c.content) > 120 else c.content,
                        "char_count": len(c.content),
                    }
                    for c in chunks[:3]  # Preview first 3 chunks
                ]
                return doc_dict

            except Exception as e:
                logger.exception(f"Ingestion failed for '{filename}': {str(e)}")
                document.status = "failed"
                document.error_message = str(e)
                # Re-raise so the caller can return an appropriate error response
                raise
