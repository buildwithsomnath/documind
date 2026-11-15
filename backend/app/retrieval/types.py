from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class SearchResult:
    """
    Standardized search result representation across all retrieval strategies
    (Vector, Keyword, and Hybrid).
    """
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int
    page_end: int
    content: str
    score: float
    retrieval_method: str  # 'vector', 'keyword', 'hybrid_rrf', etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "page_end": self.page_end,
            "content": self.content,
            "score": round(self.score, 4),
            "retrieval_method": self.retrieval_method,
            "metadata": self.metadata,
        }
