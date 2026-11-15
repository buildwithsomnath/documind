import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class CitationValidator:
    """
    Parses and verifies inline citations from LLM-generated answers against
    the ground-truth retrieved passages.
    """

    # Regex patterns supporting:
    # 1. [filename.ext, Page X] or [filename.ext, Pages X-Y]
    # 2. [Source: filename.ext | Page X]
    # 3. [Passage X] or [X]
    BRACKET_CITATION_REGEX = re.compile(
        r"\[(?:Source:\s*)?([a-zA-Z0-9_\-\.\s]+?)(?:,\s*|\s*\|\s*)Page(?:s)?\s*(\d+)(?:-(\d+))?\]",
        re.IGNORECASE
    )
    PASSAGE_NUM_REGEX = re.compile(
        r"\[(?:Passage\s*)?(\d+)\]",
        re.IGNORECASE
    )

    @classmethod
    def extract_and_verify(
        cls,
        answer_text: str,
        retrieved_passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Extracts citations from text and validates them against retrieved passages.

        :param answer_text: The LLM's generated response string.
        :param retrieved_passages: List of passage metadata dicts from prompt builder.
        :return: Dict containing validated citations list, accuracy score, and sources used.
        """
        citations: List[Dict[str, Any]] = []
        seen_keys = set()
        sources_used = set()

        # Build lookup tables from retrieved passages
        # 1. By (filename.lower(), page_num)
        # 2. By passage_index (1, 2, 3...)
        passage_by_doc_page = {}
        passage_by_index = {}

        for p in retrieved_passages:
            doc_name = p["filename"].lower().strip()
            page_num = p["page_number"]
            passage_by_doc_page[(doc_name, page_num)] = p
            passage_by_index[p.get("passage_index", 0)] = p

        # 1. Match [filename, Page X] style citations
        for match in cls.BRACKET_CITATION_REGEX.finditer(answer_text):
            raw_tag = match.group(0)
            doc_name = match.group(1).strip()
            page_num = int(match.group(2))

            key = (doc_name.lower(), page_num)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Verification check
            is_verified = key in passage_by_doc_page
            matched_passage = passage_by_doc_page.get(key, {})

            citation_data = {
                "tag": raw_tag,
                "filename": doc_name,
                "page_number": page_num,
                "verified": is_verified,
                "chunk_id": matched_passage.get("chunk_id"),
                "relevance_score": matched_passage.get("score"),
            }
            citations.append(citation_data)
            if is_verified:
                sources_used.add(doc_name)

        # 2. Match [Passage X] style citations if present
        for match in cls.PASSAGE_NUM_REGEX.finditer(answer_text):
            p_num = int(match.group(1))
            if p_num in passage_by_index:
                p_meta = passage_by_index[p_num]
                key = (p_meta["filename"].lower(), p_meta["page_number"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                citations.append({
                    "tag": match.group(0),
                    "filename": p_meta["filename"],
                    "page_number": p_meta["page_number"],
                    "verified": True,
                    "chunk_id": p_meta.get("chunk_id"),
                    "relevance_score": p_meta.get("score"),
                })
                sources_used.add(p_meta["filename"])

        total = len(citations)
        verified_count = sum(1 for c in citations if c["verified"])
        accuracy = (verified_count / total * 100.0) if total > 0 else 100.0

        return {
            "citations": citations,
            "total_citations": total,
            "verified_citations": verified_count,
            "citation_accuracy_percentage": round(accuracy, 1),
            "sources_used": sorted(list(sources_used)),
        }
