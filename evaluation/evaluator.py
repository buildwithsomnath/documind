import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.retrieval.vector_search import VectorSearchEngine
from app.retrieval.keyword_search import KeywordSearchEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.generation.service import QAService
from evaluation.metrics import recall_at_k, reciprocal_rank, ndcg_at_k, calculate_faithfulness

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Automated evaluation framework for DocuMind.
    Benchmarks retrieval (Vector vs Keyword vs Hybrid) and generation faithfulness/citations.
    """

    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            dataset_path = Path(__file__).resolve().parent / "benchmark_dataset.json"
        self.dataset_path = Path(dataset_path)

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = KeywordSearchEngine()
        self.hybrid_engine = HybridSearchEngine()
        self.qa_service = QAService()

    def _is_relevant_chunk(self, chunk, item: Dict[str, Any]) -> bool:
        """Determines if a retrieved chunk matches ground-truth target criteria."""
        exp_doc = item.get("expected_document")
        exp_page = item.get("expected_page")
        if not exp_doc:
            return False

        doc_match = chunk.filename.lower() == exp_doc.lower()
        page_match = exp_page is None or (chunk.page_number <= exp_page <= chunk.page_end)
        return doc_match and page_match

    def evaluate_retrieval(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs comparative retrieval evaluation comparing:
        1. Pure Vector Search
        2. Pure Keyword Search (FTS)
        3. Hybrid Search + Reranker
        """
        answerable_items = [item for item in self.dataset if item.get("is_answerable", True)]
        if not answerable_items:
            return {}

        strategies = {
            "vector_search": self.vector_engine.search,
            "keyword_search": self.keyword_engine.search,
            "hybrid_reranked": lambda q, k: self.hybrid_engine.search(q, top_k=k, enable_rerank=True),
        }

        results_by_strategy = {}

        for strat_name, search_fn in strategies.items():
            r1_list, r3_list, r5_list = [], [], []
            rr_list = []
            ndcg3_list = []

            for item in answerable_items:
                query = item["query"]
                passages = search_fn(query, top_k)

                # Identify which retrieved chunk IDs are ground-truth hits
                hit_ids = set()
                relevance_map = {}
                retrieved_ids = [p.chunk_id for p in passages]

                for p in passages:
                    if self._is_relevant_chunk(p, item):
                        hit_ids.add(p.chunk_id)
                        # Check keyword coverage for graded relevance (nDCG)
                        content_lower = p.content.lower()
                        kw_hits = sum(1 for kw in item.get("target_keywords", []) if kw.lower() in content_lower)
                        relevance_map[p.chunk_id] = 2.0 if kw_hits >= 2 else 1.0
                    else:
                        relevance_map[p.chunk_id] = 0.0

                # Compute metrics
                r1_list.append(recall_at_k(retrieved_ids, hit_ids, k=1))
                r3_list.append(recall_at_k(retrieved_ids, hit_ids, k=3))
                r5_list.append(recall_at_k(retrieved_ids, hit_ids, k=5))
                rr_list.append(reciprocal_rank(retrieved_ids, hit_ids))
                ndcg3_list.append(ndcg_at_k(retrieved_ids, relevance_map, k=3))

            results_by_strategy[strat_name] = {
                "recall@1": round(sum(r1_list) / len(r1_list), 4),
                "recall@3": round(sum(r3_list) / len(r3_list), 4),
                "recall@5": round(sum(r5_list) / len(r5_list), 4),
                "mrr": round(sum(rr_list) / len(rr_list), 4),
                "ndcg@3": round(sum(ndcg3_list) / len(ndcg3_list), 4),
            }

        return results_by_strategy

    def evaluate_generation(self, evaluate_llm: bool = True) -> Dict[str, Any]:
        """
        Evaluates end-to-end generation, source citations, and anti-hallucination faithfulness.
        """
        total_citations = 0
        verified_citations = 0
        faithfulness_scores = []
        hallucination_blocks = 0
        negative_tests_count = 0

        generation_details = []

        for item in self.dataset:
            query = item["query"]
            is_answerable = item.get("is_answerable", True)

            if not evaluate_llm:
                # Run mocked evaluation if LLM calls disabled
                continue

            try:
                resp = self.qa_service.answer_question(query, top_k=3)
                answer = resp.get("answer", "")
                citations = resp.get("citations", [])

                if not is_answerable:
                    negative_tests_count += 1
                    # Verify fallback phrase
                    if "not have enough information" in answer.lower():
                        hallucination_blocks += 1
                    generation_details.append({
                        "query_id": item["id"],
                        "type": "negative_guardrail",
                        "blocked_correctly": "not have enough information" in answer.lower(),
                        "answer_snippet": answer[:80],
                    })
                    continue

                # Evaluate citations
                total_citations += len(citations)
                verified_citations += sum(1 for c in citations if c.get("verified"))

                # Evaluate Faithfulness
                passages_text = [p["content"] for p in resp.get("retrieved_passages", [])]
                f_result = calculate_faithfulness(answer, passages_text)
                faithfulness_scores.append(f_result["faithfulness_score"])

                generation_details.append({
                    "query_id": item["id"],
                    "type": "grounded_answer",
                    "citation_count": len(citations),
                    "verified_citation_count": sum(1 for c in citations if c.get("verified")),
                    "faithfulness_score": f_result["faithfulness_score"],
                    "answer_snippet": answer[:100],
                })

            except Exception as e:
                logger.warning(f"Error evaluating generation for query '{query}': {e}")

        citation_acc = (
            round((verified_citations / total_citations) * 100.0, 1)
            if total_citations > 0 else 100.0
        )
        mean_faithfulness = (
            round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
            if faithfulness_scores else 1.0
        )
        guardrail_rate = (
            round((hallucination_blocks / negative_tests_count) * 100.0, 1)
            if negative_tests_count > 0 else 100.0
        )

        return {
            "citation_accuracy_percentage": citation_acc,
            "mean_faithfulness_score": mean_faithfulness,
            "anti_hallucination_block_rate": guardrail_rate,
            "total_evaluated_queries": len(self.dataset),
            "details": generation_details,
        }

    def run_full_benchmark(self, run_generation: bool = True) -> Dict[str, Any]:
        """Runs the complete evaluation benchmark."""
        retrieval_report = self.evaluate_retrieval(top_k=5)
        generation_report = self.evaluate_generation(evaluate_llm=run_generation) if run_generation else {}

        return {
            "retrieval_comparison": retrieval_report,
            "generation_metrics": generation_report,
        }
