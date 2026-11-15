import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, DateTime, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.db import Base
from app.config import Config


class DocumentChunk(Base):
    """
    Represents a single text chunk extracted from a document.
    Stores chunk text, page boundary metadata, and 384-dimensional vector embedding.
    """
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    page_end = Column(Integer, nullable=False)
    # Using 'metadata' as the column name in SQL, mapped to chunk_metadata attribute
    # to avoid conflicting with SQLAlchemy's reserved Base.metadata attribute
    chunk_metadata = Column("metadata", JSONB, default=dict, nullable=False)
    embedding = Column(Vector(Config.EMBEDDING_DIM), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_document_chunks_doc_chunk_idx", "document_id", "chunk_index"),
        # HNSW vector index using cosine distance
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )

    def to_dict(self, include_embedding=False):
        data = {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_index": self.chunk_index,
            "content": self.content,
            "page_number": self.page_number,
            "page_end": self.page_end,
            "metadata": self.chunk_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        if include_embedding and self.embedding is not None:
            data["embedding_dim"] = len(self.embedding)
        return data
