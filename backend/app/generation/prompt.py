import logging
from typing import List, Tuple
from app.retrieval.types import SearchResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are DocuMind, an enterprise document intelligence and question-answering assistant.
Your task is to provide clear, factual, and strictly grounded answers to the user's questions based EXCLUSIVELY on the provided reference passages.

MANDATORY GROUNDING RULES:
1. Answer ONLY using the facts explicitly stated in the reference passages below.
2. Do NOT extrapolate, speculate, or incorporate outside knowledge not contained in the passages.
3. If the reference passages do not contain enough facts to answer the question, you MUST respond with:
   "I do not have enough information in the provided documents to answer this question."
4. Every factual assertion must be attributed to its source using bracketed citations containing the document name and page number, for example: [sample_architecture.pdf, Page 2].
5. Keep your tone objective, concise, and professional."""


def format_retrieved_context(passages: List[SearchResult]) -> Tuple[str, List[dict]]:
    """
    Formats a list of retrieved passages into a structured text block with clear provenance headers.

    :param passages: Ranked list of SearchResult objects.
    :return: Tuple of (formatted_context_string, list_of_passage_metadata).
    """
    if not passages:
        return "No relevant passages found.", []

    context_blocks = []
    passage_references = []

    for idx, p in enumerate(passages, start=1):
        # Format page span: e.g. "Page 2" or "Pages 2-3"
        page_str = f"Page {p.page_number}" if p.page_number == p.page_end else f"Pages {p.page_number}-{p.page_end}"
        header = f"[Passage {idx}] Source: {p.filename} | {page_str} (Relevance Score: {p.score})"
        block = f"{header}\n{p.content}"
        context_blocks.append(block)

        passage_references.append({
            "passage_index": idx,
            "chunk_id": p.chunk_id,
            "document_id": p.document_id,
            "filename": p.filename,
            "page_number": p.page_number,
            "page_end": p.page_end,
            "score": p.score,
        })

    formatted_context = "\n\n---\n\n".join(context_blocks)
    return formatted_context, passage_references


def build_grounded_prompt(question: str, passages: List[SearchResult]) -> Tuple[str, str, List[dict]]:
    """
    Assembles the final system prompt and user prompt for the LLM.

    :param question: The user's natural language question.
    :param passages: Retrieved passages from the hybrid search engine.
    :return: Tuple of (system_prompt, user_prompt, passage_references).
    """
    context_text, passage_refs = format_retrieved_context(passages)

    user_prompt = f"""REFERENCE PASSAGES:
----------------------------------------
{context_text}
----------------------------------------

USER QUESTION:
{question}

INSTRUCTIONS:
Provide a comprehensive answer to the question above using ONLY the reference passages. Include inline citations such as [DocumentName, Page X] for every claim made."""

    return SYSTEM_PROMPT, user_prompt, passage_refs
