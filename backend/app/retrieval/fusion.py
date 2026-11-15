import logging
from typing import List, Dict, Any
from app.config import Config
from app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_results: List[SearchResult],
    keyword_results: List[SearchResult],
    rrf_k: int = Config.RRF_K,
    top_n: int = 20,
) -> List[SearchResult]:
    """
    Combines ranked lists from semantic vector search and lexical keyword search
    using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in {vector, keyword}} (1 / (k + rank_m(d)))

    :param vector_results: Ranked list of SearchResult from VectorSearchEngine.
    :param keyword_results: Ranked list of SearchResult from KeywordSearchEngine.
    :param rrf_k: Smoothing constant (default: 60) mitigating ranking extremes.
    :param top_n: Number of fused candidate results to return.
    :return: Merged, deduplicated list of SearchResult ordered by RRF score descending.
    """
    # Map chunk_id -> { "result": SearchResult, "rrf_score": float, "vector_rank": int, "keyword_rank": int }
    candidates: Dict[str, Dict[str, Any]] = {}

    # Process vector rankings (1-indexed)
    for rank, res in enumerate(vector_results, start=1):
        chunk_id = res.chunk_id
        weight = 1.0 / (rrf_k + rank)

        candidates[chunk_id] = {
            "result": res,
            "rrf_score": weight,
            "vector_rank": rank,
            "vector_score": res.score,
            "keyword_rank": None,
            "keyword_score": None,
        }

    # Process keyword rankings (1-indexed)
    for rank, res in enumerate(keyword_results, start=1):
        chunk_id = res.chunk_id
        weight = 1.0 / (rrf_k + rank)

        if chunk_id in candidates:
            # Chunk retrieved by BOTH search methods: accumulate score
            candidates[chunk_id]["rrf_score"] += weight
            candidates[chunk_id]["keyword_rank"] = rank
            candidates[chunk_id]["keyword_score"] = res.score
        else:
            candidates[chunk_id] = {
                "result": res,
                "rrf_score": weight,
                "vector_rank": None,
                "vector_score": None,
                "keyword_rank": rank,
                "keyword_score": res.score,
            }

    # Sort descending by RRF score
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    fused_results: List[SearchResult] = []
    for item in sorted_candidates[:top_n]:
        orig = item["result"]
        # Merge provenance metadata
        metadata = dict(orig.metadata or {})
        metadata["rrf_provenance"] = {
            "rrf_score": round(item["rrf_score"], 6),
            "vector_rank": item["vector_rank"],
            "vector_score": item["vector_score"],
            "keyword_rank": item["keyword_rank"],
            "keyword_score": item["keyword_score"],
            "found_in_both": item["vector_rank"] is not None and item["keyword_rank"] is not None,
        }

        fused_results.append(
            SearchResult(
                chunk_id=orig.chunk_id,
                document_id=orig.document_id,
                filename=orig.filename,
                chunk_index=orig.chunk_index,
                page_number=orig.page_number,
                page_end=orig.page_end,
                content=orig.content,
                score=round(item["rrf_score"], 6),
                retrieval_method="hybrid_rrf",
                metadata=metadata,
            )
        )

    logger.info(
        f"RRF fused {len(vector_results)} vector and {len(keyword_results)} keyword results "
        f"into {len(fused_results)} candidates (top_n={top_n})"
    )
    return fused_results
