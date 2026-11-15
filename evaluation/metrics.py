import math
from typing import List, Set, Dict, Any


def recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """
    Computes Recall@K:
    Fraction of ground-truth relevant items successfully retrieved in the top K positions.

    Formula:
        Recall@K = |Retrieved@K ∩ GroundTruth| / |GroundTruth|
    """
    if not ground_truth_ids:
        return 1.0 if not retrieved_ids else 0.0

    top_k_retrieved = set(retrieved_ids[:k])
    hits = len(top_k_retrieved.intersection(ground_truth_ids))
    return round(hits / len(ground_truth_ids), 4)


def reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """
    Computes Reciprocal Rank (RR):
    1 / rank of the FIRST relevant item found. Returns 0.0 if no relevant items retrieved.

    Formula:
        RR = 1 / rank_first_hit
    """
    if not ground_truth_ids:
        return 0.0

    for idx, item_id in enumerate(retrieved_ids, start=1):
        if item_id in ground_truth_ids:
            return round(1.0 / idx, 4)
    return 0.0


def mean_reciprocal_rank(queries_retrieved: List[List[str]], queries_ground_truth: List[Set[str]]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) across a collection of queries.

    Formula:
        MRR = (1 / |Q|) * sum_{q=1}^{|Q|} RR_q
    """
    if not queries_retrieved or not queries_ground_truth:
        return 0.0

    total_rr = sum(
        reciprocal_rank(ret, gt)
        for ret, gt in zip(queries_retrieved, queries_ground_truth)
    )
    return round(total_rr / len(queries_retrieved), 4)


def ndcg_at_k(
    retrieved_ids: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (nDCG@K).
    Evaluates both relevance quality and the positional ordering of results.

    Formula:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
        IDCG@K = Ideal DCG (sorted descending by relevance)
        nDCG@K = DCG@K / IDCG@K
    """
    if not relevance_scores or k <= 0:
        return 0.0

    # 1. Compute Discounted Cumulative Gain (DCG@K)
    dcg = 0.0
    for idx, item_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(item_id, 0.0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(idx + 1.0)
        dcg += gain / discount

    # 2. Compute Ideal Discounted Cumulative Gain (IDCG@K)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for idx, rel in enumerate(ideal_scores, start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(idx + 1.0)
        idcg += gain / discount

    if idcg == 0.0:
        return 0.0

    return round(min(1.0, dcg / idcg), 4)


def calculate_faithfulness(answer: str, context_passages: List[str]) -> Dict[str, Any]:
    """
    Evaluates answer faithfulness (groundedness / hallucination detection).
    Splits answer into individual factual sentences and verifies whether the
    factual claims are supported by the retrieved context.

    Returns:
        faithfulness_score: Fraction in [0.0, 1.0]
        supported_sentences: Count of supported sentences
        total_sentences: Count of total factual sentences
    """
    if not answer or not context_passages:
        return {"faithfulness_score": 0.0, "supported_sentences": 0, "total_sentences": 0}

    # Normalize context into a single searchable text
    combined_context = " ".join(context_passages).lower()

    # Split answer into sentences
    import re
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", answer)
        if len(s.strip()) > 15
    ]

    if not sentences:
        return {"faithfulness_score": 1.0, "supported_sentences": 0, "total_sentences": 0}

    supported_count = 0
    for sent in sentences:
        # Extract meaningful keywords (words > 3 characters, excluding stop words)
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{4,}\b", sent)]
        if not words:
            supported_count += 1
            continue

        # Check keyword density in combined context
        matched_words = [w for w in words if w in combined_context]
        overlap_ratio = len(matched_words) / len(words)

        # Sentence is considered supported if >= 65% of its key terms appear in context
        if overlap_ratio >= 0.65:
            supported_count += 1

    score = round(supported_count / len(sentences), 4)
    return {
        "faithfulness_score": score,
        "supported_sentences": supported_count,
        "total_sentences": len(sentences),
    }
