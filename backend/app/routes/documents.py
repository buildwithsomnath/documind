import logging
import uuid
from flask import Blueprint, request, jsonify
from app.db import get_db_session
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

documents_bp = Blueprint("documents", __name__, url_prefix="/api/documents")


@documents_bp.post("/upload")
def upload_document():
    """
    Endpoint to upload and ingest a PDF or DOCX document.
    Extracts text, generates overlapping chunks, computes 384-d vector embeddings,
    and stores everything in PostgreSQL via pgvector.

    Accepts:
        multipart/form-data:
            - file: The PDF or DOCX file
            - chunk_size (optional): Override character chunk size
            - chunk_overlap (optional): Override character chunk overlap
    """
    if "file" not in request.files:
        return jsonify({
            "error": "Bad Request",
            "message": "Missing 'file' field in multipart/form-data."
        }), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({
            "error": "Bad Request",
            "message": "No file selected for upload."
        }), 400

    # Optional chunking overrides from form data
    chunk_size = request.form.get("chunk_size", type=int)
    chunk_overlap = request.form.get("chunk_overlap", type=int)

    pipeline_kwargs = {}
    if chunk_size:
        pipeline_kwargs["chunk_size"] = chunk_size
    if chunk_overlap:
        pipeline_kwargs["chunk_overlap"] = chunk_overlap

    try:
        pipeline = IngestionPipeline(**pipeline_kwargs)
        result = pipeline.process_file_storage(file)

        return jsonify({
            "status": "success",
            "message": f"Document '{result['filename']}' successfully ingested.",
            "document": result
        }), 201

    except ValueError as ve:
        logger.warning(f"Validation error during upload: {ve}")
        return jsonify({
            "error": "Validation Error",
            "message": str(ve)
        }), 400

    except Exception as e:
        logger.exception("Unexpected error during document ingestion")
        return jsonify({
            "error": "Internal Server Error",
            "message": f"Failed to ingest document: {str(e)}"
        }), 500


@documents_bp.get("")
def list_documents():
    """
    Retrieves all ingested documents with metadata and status.
    """
    with get_db_session() as session:
        documents = session.query(Document).order_by(Document.created_at.desc()).all()
        return jsonify({
            "total_documents": len(documents),
            "documents": [doc.to_dict() for doc in documents]
        }), 200


@documents_bp.get("/<uuid:document_id>")
def get_document(document_id: uuid.UUID):
    """
    Retrieves a single document along with its chunk details and provenance.
    """
    include_chunks = request.args.get("include_chunks", "true").lower() == "true"

    with get_db_session() as session:
        document = session.query(Document).filter(Document.id == document_id).first()
        if not document:
            return jsonify({
                "error": "Not Found",
                "message": f"Document with ID '{document_id}' was not found."
            }), 404

        doc_data = document.to_dict()

        if include_chunks:
            chunks = (
                session.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
                .all()
            )
            doc_data["chunks"] = [
                chunk.to_dict(include_embedding=True)
                for chunk in chunks
            ]

        return jsonify(doc_data), 200


@documents_bp.delete("/<uuid:document_id>")
def delete_document(document_id: uuid.UUID):
    """
    Deletes a document and its cascading chunks and vector embeddings.
    """
    with get_db_session() as session:
        document = session.query(Document).filter(Document.id == document_id).first()
        if not document:
            return jsonify({
                "error": "Not Found",
                "message": f"Document with ID '{document_id}' was not found."
            }), 404

        session.delete(document)
        return jsonify({
            "status": "success",
            "message": f"Document '{document.filename}' ({document_id}) and its chunks deleted."
        }), 200
