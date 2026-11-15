import logging
import uuid
from typing import List, Optional
from sqlalchemy import func

from app.db import get_db_session
from app.models.document import Document
from app.models.chunk import DocumentChunk
from app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)


class KeywordSearchEngine:
    """
    Executes lexical / keyword searches over document chunks using PostgreSQL Full-Text Search (FTS).
    Uses websearch_to_tsquery (supporting phrases and boolean logic) and ts_rank_cd (Cover Density).
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[uuid.UUID] = None,
    ) -> List[SearchResult]:
        """
        Searches document chunks for lexical keyword matches using PostgreSQL Full-Text Search.

        :param query: Keyword search phrase or terms.
        :param top_k: Maximum number of relevant chunks to retrieve.
        :param document_id: Optional UUID to constrain search to a single document.
        :return: Ranked list of SearchResult objects ordered by FTS rank descending.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()

        with get_db_session() as session:
            # Build PostgreSQL Full-Text Search vector and query expressions
            ts_vector = func.to_tsvector("english", DocumentChunk.content)
            ts_query = func.websearch_to_tsquery("english", clean_query)
            rank_expr = func.ts_rank_cd(ts_vector, ts_query)

            query_stmt = (
                session.query(
                    DocumentChunk,
                    Document.filename,
                    rank_expr.label("rank")
                )
                .join(Document, DocumentChunk.document_id == Document.id)
                # @@ operator performs full-text matching: tsvector @@ tsquery
                .filter(ts_vector.op("@@")(ts_query))
            )

            if document_id:
                query_stmt = query_stmt.filter(DocumentChunk.document_id == document_id)

            results = (
                query_stmt
                .order_by(rank_expr.desc())
                .limit(top_k)
                .all()
            )

            search_results: List[SearchResult] = []
            for chunk, filename, rank in results:
                search_results.append(
                    SearchResult(
                        chunk_id=str(chunk.id),
                        document_id=str(chunk.document_id),
                        filename=filename,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        page_end=chunk.page_end,
                        content=chunk.content,
                        score=float(rank),
                        retrieval_method="keyword",
                        metadata=chunk.chunk_metadata or {},
                    )
                )

            logger.info(
                f"Keyword search for '{clean_query[:40]}' returned {len(search_results)} chunk(s) (top_k={top_k})"
            )
            return search_results
