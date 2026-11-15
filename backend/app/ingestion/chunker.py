import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any
from app.ingestion.extractor import ExtractedPage

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    chunk_index: int
    content: str
    page_number: int  # Starting page
    page_end: int     # Ending page
    metadata: Dict[str, Any] = field(default_factory=dict)


class PageAwareChunker:
    """
    Splits extracted document pages into overlapping text chunks while:
    1. Preserving exact page provenance for source citations.
    2. Respecting sentence and paragraph boundaries to preserve semantic coherence.
    3. Ensuring chunks fit within the token context window of sentence-transformers/all-MiniLM-L6-v2 (256 tokens).
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        :param chunk_size: Target maximum chunk length in characters (~100-120 words).
        :param chunk_overlap: Overlap length in characters to retain context across boundaries.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """
        Splits text into sentences based on punctuation (. ! ?) followed by whitespace,
        or newline boundaries.
        """
        # Split on double newlines (paragraphs) first, then sentences
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: List[str] = []

        for para in paragraphs:
            # Split sentences within paragraph
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for s in sentences:
                s = s.strip()
                if s:
                    units.append(s)

        return units

    def _chunk_page_text(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """
        Chunks text within a single page using sentence-level boundary packing with overlap.
        """
        if not text or len(text.strip()) == 0:
            return []

        # If the entire page is already smaller than chunk_size, return it as a single chunk
        if len(text) <= self.chunk_size:
            return [{
                "content": text.strip(),
                "page_number": page_number,
                "page_end": page_number,
                "char_count": len(text.strip()),
                "word_count": len(text.strip().split()),
            }]

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[Dict[str, Any]] = []
        current_chunk_sentences: List[str] = []
        current_len = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]

            # If a single sentence exceeds chunk_size, split it with sliding character window
            if len(sentence) > self.chunk_size:
                # Flush existing buffer first
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences).strip()
                    if chunk_text:
                        chunks.append({
                            "content": chunk_text,
                            "page_number": page_number,
                            "page_end": page_number,
                            "char_count": len(chunk_text),
                            "word_count": len(chunk_text.split()),
                        })
                    current_chunk_sentences = []
                    current_len = 0

                # Split long sentence by character sliding window
                start = 0
                step = self.chunk_size - self.chunk_overlap
                while start < len(sentence):
                    end = min(start + self.chunk_size, len(sentence))
                    # Avoid cutting words in half if possible
                    if end < len(sentence) and " " in sentence[start:end]:
                        last_space = sentence.rfind(" ", start, end)
                        if last_space > start:
                            end = last_space

                    slice_text = sentence[start:end].strip()
                    if slice_text:
                        chunks.append({
                            "content": slice_text,
                            "page_number": page_number,
                            "page_end": page_number,
                            "char_count": len(slice_text),
                            "word_count": len(slice_text.split()),
                        })
                    start += step

                i += 1
                continue

            # Check if adding this sentence exceeds chunk_size
            projected_len = current_len + (1 if current_chunk_sentences else 0) + len(sentence)

            if projected_len <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_len = projected_len
                i += 1
            else:
                # Emit current chunk
                chunk_text = " ".join(current_chunk_sentences).strip()
                if chunk_text:
                    chunks.append({
                        "content": chunk_text,
                        "page_number": page_number,
                        "page_end": page_number,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                    })

                # Compute overlap for the next chunk:
                # Keep sentences from the tail of current_chunk_sentences that sum up to ~chunk_overlap
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev_sent in reversed(current_chunk_sentences):
                    if overlap_len + len(prev_sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_len += len(prev_sent) + 1
                    else:
                        break

                current_chunk_sentences = list(overlap_sentences)
                current_len = sum(len(s) for s in current_chunk_sentences) + max(0, len(current_chunk_sentences) - 1)

                # Add the pending sentence to the fresh buffer
                current_chunk_sentences.append(sentence)
                current_len += (1 if current_len > 0 else 0) + len(sentence)
                i += 1

        # Emit any remaining text in buffer
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text and (not chunks or chunk_text != chunks[-1]["content"]):
                chunks.append({
                    "content": chunk_text,
                    "page_number": page_number,
                    "page_end": page_number,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                })

        return chunks

    def chunk_pages(self, pages: List[ExtractedPage]) -> List[TextChunk]:
        """
        Takes a list of ExtractedPage objects and returns indexed TextChunk objects.
        """
        all_chunks: List[TextChunk] = []
        global_chunk_index = 0

        for page in pages:
            page_chunks = self._chunk_page_text(page.text, page.page_number)

            for chunk_data in page_chunks:
                metadata = {
                    "page_number": chunk_data["page_number"],
                    "page_end": chunk_data["page_end"],
                    "char_count": chunk_data["char_count"],
                    "word_count": chunk_data["word_count"],
                    "source_page_metadata": page.metadata,
                }

                all_chunks.append(
                    TextChunk(
                        chunk_index=global_chunk_index,
                        content=chunk_data["content"],
                        page_number=chunk_data["page_number"],
                        page_end=chunk_data["page_end"],
                        metadata=metadata,
                    )
                )
                global_chunk_index += 1

        logger.info(
            f"Chunked {len(pages)} page(s) into {len(all_chunks)} chunk(s) "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks
