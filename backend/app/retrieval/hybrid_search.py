import logging
import uuid
from typing import List, Optional

from app.config import Config
from app.retrieval.types import SearchResult
from app.retrieval.vector_search import VectorSearchEngine
from app.retrieval.keyword_search import KeywordSearchEngine
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import RerankerService

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Two-Stage Hybrid Retrieval Engine:
      Stage 1 (Retrieval): Concurrent / combined Dense Vector Search + Lexical Full-Text Search.
      Stage 1.5 (Fusion): Reciprocal Rank Fusion (RRF) combines both ranked lists.
      Stage 2 (Reranking): Cross-Encoder scores the top candidates using full cross-attention.
    """

    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = KeywordSearchEngine()
        self.reranker = RerankerService.get_instance()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[uuid.UUID] = None,
        enable_rerank: bool = True,
        candidate_k: int = 20,
    ) -> List[SearchResult]:
        """
        Executes end-to-end hybrid retrieval with optional cross-encoder reranking.

        :param query: Natural language search question.
        :param top_k: Final number of passages to return.
        :param document_id: Optional document UUID to constrain retrieval scope.
        :param enable_rerank: Whether to apply Cross-Encoder reranking to candidate pool.
        :param candidate_k: Number of candidates to retrieve from each Stage 1 engine.
        :return: Final ranked list of SearchResult passages.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()

        # Stage 1: Retrieve candidate pools from both engines
        logger.info(f"Stage 1: Fetching candidates (candidate_k={candidate_k}) for query: '{clean_query[:40]}'...")
        vector_candidates = self.vector_engine.search(
            query=clean_query,
            top_k=candidate_k,
            document_id=document_id
        )

        keyword_candidates = self.keyword_engine.search(
            query=clean_query,
            top_k=candidate_k,
            document_id=document_id
        )

        # Stage 1.5: Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(
            vector_results=vector_candidates,
            keyword_results=keyword_candidates,
            rrf_k=Config.RRF_K,
            top_n=candidate_k,
        )

        if not fused_candidates:
            logger.info("Hybrid retrieval found 0 matching candidates.")
            return []

        # Stage 2: Cross-Encoder Reranking
        if enable_rerank:
            try:
                logger.info(f"Stage 2: Reranking {len(fused_candidates)} candidates via Cross-Encoder...")
                final_results = self.reranker.rerank(
                    query=clean_query,
                    candidates=fused_candidates,
                    top_k=top_k
                )
                return final_results
            except Exception as e:
                logger.warning(
                    f"Reranking encountered an error ({e}). Gracefully falling back to RRF candidates."
                )
                return fused_candidates[:top_k]

        return fused_candidates[:top_k]
