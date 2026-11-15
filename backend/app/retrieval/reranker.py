import logging
import math
import os
import threading
from typing import List, Optional
from sentence_transformers import CrossEncoder

from app.config import Config
from app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function mapping raw model logits to [0.0, 1.0].
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class RerankerService:
    """
    Singleton service that wraps a Cross-Encoder transformer model to rerank candidate passages.
    Cross-encoders process query and chunk together with full cross-attention between all tokens,
    yielding significantly higher ranking precision than bi-encoder embeddings alone.
    """
    _instance: Optional["RerankerService"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or Config.RERANKER_MODEL_NAME
        self._model: Optional[CrossEncoder] = None
        self._load_lock = threading.Lock()

    @property
    def model(self) -> CrossEncoder:
        """
        Lazily loads the cross-encoder model on the first reranking call.
        """
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info(f"Loading Cross-Encoder model '{self.model_name}' into memory...")
                    hf_token = os.getenv("HF_TOKEN") or False
                    self._model = CrossEncoder(self.model_name, token=hf_token)
                    logger.info("Cross-Encoder model loaded successfully.")
        return self._model

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "RerankerService":
        """
        Thread-safe singleton accessor.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name)
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Scores and reorders a candidate pool of SearchResult objects using full cross-attention.

        :param query: Natural language user query.
        :param candidates: List of SearchResult objects (e.g. from RRF).
        :param top_k: Maximum number of reranked results to return.
        :return: Reordered list of SearchResult objects sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        if not query or not query.strip():
            return candidates[:top_k]

        clean_query = query.strip()
        # Form [query, passage] pairs for cross-encoder inference
        pairs = [[clean_query, c.content] for c in candidates]

        logger.info(f"Cross-encoder scoring {len(pairs)} candidate passages...")
        raw_scores = self.model.predict(pairs)

        # Convert numpy or list of scores to floats
        if hasattr(raw_scores, "tolist"):
            scores_list = raw_scores.tolist()
        else:
            scores_list = list(raw_scores)

        scored_candidates = []
        for idx, (candidate, raw_score) in enumerate(zip(candidates, scores_list)):
            score_float = float(raw_score)
            normalized_score = round(sigmoid(score_float), 4)

            # Record reranking provenance
            metadata = dict(candidate.metadata or {})
            metadata["rerank_provenance"] = {
                "raw_logit": round(score_float, 4),
                "normalized_confidence": normalized_score,
                "initial_candidate_rank": idx + 1,
                "initial_score": candidate.score,
                "initial_method": candidate.retrieval_method,
            }

            reranked_result = SearchResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                filename=candidate.filename,
                chunk_index=candidate.chunk_index,
                page_number=candidate.page_number,
                page_end=candidate.page_end,
                content=candidate.content,
                score=normalized_score,
                retrieval_method="hybrid_reranked",
                metadata=metadata,
            )
            scored_candidates.append(reranked_result)

        # Sort descending by cross-encoder relevance score
        scored_candidates.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            f"Reranking complete. Selected top {min(top_k, len(scored_candidates))} results."
        )
        return scored_candidates[:top_k]
