import logging
import uuid
from typing import List, Optional

from app.db import get_db_session
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.ingestion.embedder import EmbeddingService
from app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """
    Executes dense semantic vector searches over document chunks using pgvector.
    Uses cosine distance (<=>) accelerated by the HNSW index.
    """

    def __init__(self):
        self.embedder = EmbeddingService.get_instance()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[uuid.UUID] = None,
    ) -> List[SearchResult]:
        """
        Searches document chunks semantically matching the query.

        :param query: Natural language search string.
        :param top_k: Maximum number of relevant chunks to retrieve.
        :param document_id: Optional UUID to constrain search to a single document.
        :return: Ranked list of SearchResult objects ordered by cosine similarity descending.
        """
        if not query or not query.strip():
            return []

        # 1. Generate 384-dimensional normalized embedding for query
        query_vector = self.embedder.embed_query(query.strip())

        # 2. Query PostgreSQL with pgvector cosine distance
        with get_db_session() as session:
            # DocumentChunk.embedding.cosine_distance() compiles to embedding <=> :query_vector
            distance_expr = DocumentChunk.embedding.cosine_distance(query_vector)

            query_stmt = (
                session.query(
                    DocumentChunk,
                    Document.filename,
                    distance_expr.label("distance")
                )
                .join(Document, DocumentChunk.document_id == Document.id)
            )

            # Optional filter by document_id
            if document_id:
                query_stmt = query_stmt.filter(DocumentChunk.document_id == document_id)

            # Cosine distance: 0.0 means identical, 2.0 means opposite
            # Order ascending by distance (most similar first)
            results = (
                query_stmt
                .order_by(distance_expr.asc())
                .limit(top_k)
                .all()
            )

            # 3. Format results into standardized SearchResult objects
            search_results: List[SearchResult] = []
            for chunk, filename, distance in results:
                # For normalized unit vectors: Cosine Similarity = 1.0 - Cosine Distance
                # Clamp similarity between 0.0 and 1.0 for clean scoring
                similarity = max(0.0, min(1.0, 1.0 - float(distance)))

                search_results.append(
                    SearchResult(
                        chunk_id=str(chunk.id),
                        document_id=str(chunk.document_id),
                        filename=filename,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        page_end=chunk.page_end,
                        content=chunk.content,
                        score=similarity,
                        retrieval_method="vector",
                        metadata=chunk.chunk_metadata or {},
                    )
                )

            logger.info(
                f"Vector search for '{query[:40]}' returned {len(search_results)} chunk(s) (top_k={top_k})"
            )
            return search_results
