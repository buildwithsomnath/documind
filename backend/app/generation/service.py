import logging
import uuid
from typing import Dict, Any, Optional

from app.retrieval.hybrid_search import HybridSearchEngine
from app.generation.prompt import build_grounded_prompt
from app.generation.gemini_client import GeminiClient
from app.generation.citations import CitationValidator

logger = logging.getLogger(__name__)


class QAService:
    """
    End-to-End Grounded Q&A Service:
    Hybrid Retrieval -> Grounded Prompting -> Gemini Generation -> Citation Verification.
    """

    def __init__(self):
        self.hybrid_engine = HybridSearchEngine()
        self.gemini_client = GeminiClient()

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        document_id: Optional[uuid.UUID] = None,
        enable_rerank: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline for a user question.

        :param question: The natural language question.
        :param top_k: Number of top passages to supply to the LLM.
        :param document_id: Optional document UUID filter.
        :param enable_rerank: Whether to enable cross-encoder reranking.
        :return: Dict containing answer, verified citations, metrics, and retrieved passages.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        clean_question = question.strip()

        # 1. Retrieve top passages using Two-Stage Hybrid Search
        logger.info(f"Retrieving top {top_k} passages for question: '{clean_question[:40]}'...")
        passages = self.hybrid_engine.search(
            query=clean_question,
            top_k=top_k,
            document_id=document_id,
            enable_rerank=enable_rerank,
        )

        # 2. Check for empty retrieval
        if not passages:
            return {
                "question": clean_question,
                "answer": "I do not have enough information in the provided documents to answer this question.",
                "citations": [],
                "sources_used": [],
                "citation_accuracy_percentage": 100.0,
                "retrieved_passages": [],
            }

        # 3. Assemble grounded prompt
        system_prompt, user_prompt, passage_refs = build_grounded_prompt(
            question=clean_question,
            passages=passages
        )

        # 4. Generate answer using Google Gemini
        answer = self.gemini_client.generate(
            system_instruction=system_prompt,
            prompt=user_prompt
        )

        # 5. Extract and verify citations against retrieved passages
        citation_analysis = CitationValidator.extract_and_verify(
            answer_text=answer,
            retrieved_passages=passage_refs
        )

        return {
            "question": clean_question,
            "answer": answer,
            "citations": citation_analysis["citations"],
            "sources_used": citation_analysis["sources_used"],
            "total_citations": citation_analysis["total_citations"],
            "verified_citations": citation_analysis["verified_citations"],
            "citation_accuracy_percentage": citation_analysis["citation_accuracy_percentage"],
            "retrieved_passages": [p.to_dict() for p in passages],
        }
