import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import docx
import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    page_number: int  # 1-indexed
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    filename: str
    file_type: str  # 'pdf' or 'docx'
    total_pages: int
    pages: List[ExtractedPage]
    total_characters: int
    total_words: int


def clean_text(text: str) -> str:
    """
    Cleans raw extracted text:
    - Normalizes Unicode spaces and control characters
    - Fixes hyphenated line breaks (e.g. 'informa-\ntion' -> 'information')
    - Consolidates excessive consecutive blank lines while preserving paragraph boundaries
    """
    if not text:
        return ""
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated words split across lines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces and tabs into single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


class DocumentExtractor:
    """
    Extracts text and page/paragraph-level metadata from PDF and DOCX files.
    """

    @classmethod
    def extract_pdf(cls, file_source: Union[str, Path, bytes, io.BytesIO], filename: str) -> ExtractedDocument:
        """
        Extracts page-by-page text from a PDF file using PyMuPDF.
        """
        if isinstance(file_source, (str, Path)):
            doc = pymupdf.open(str(file_source))
        elif isinstance(file_source, bytes):
            doc = pymupdf.open(stream=file_source, filetype="pdf")
        elif isinstance(file_source, io.BytesIO):
            doc = pymupdf.open(stream=file_source.getvalue(), filetype="pdf")
        else:
            raise ValueError(f"Unsupported file source type: {type(file_source)}")

        try:
            if doc.is_encrypted:
                raise ValueError("Password-protected or encrypted PDFs are not supported.")

            pages: List[ExtractedPage] = []
            total_chars = 0
            total_words = 0

            for page_index in range(len(doc)):
                page = doc[page_index]
                raw_text = page.get_text("text")
                cleaned = clean_text(raw_text)
                page_num = page_index + 1

                word_count = len(cleaned.split()) if cleaned else 0
                char_count = len(cleaned)

                total_chars += char_count
                total_words += word_count

                pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        text=cleaned,
                        metadata={
                            "page_number": page_num,
                            "char_count": char_count,
                            "word_count": word_count,
                            "width": page.rect.width,
                            "height": page.rect.height,
                        }
                    )
                )

            if total_words == 0:
                logger.warning(
                    f"PDF '{filename}' yielded 0 extractable words. "
                    "The document might be scanned/image-only (requires OCR)."
                )

            return ExtractedDocument(
                filename=filename,
                file_type="pdf",
                total_pages=len(doc),
                pages=pages,
                total_characters=total_chars,
                total_words=total_words,
            )
        finally:
            doc.close()

    @classmethod
    def extract_docx(cls, file_source: Union[str, Path, bytes, io.BytesIO], filename: str) -> ExtractedDocument:
        """
        Extracts text from a DOCX file using python-docx.
        Preserves paragraph order, headings, and tracks paragraph/page approximations.
        """
        if isinstance(file_source, (str, Path)):
            doc = docx.Document(str(file_source))
        elif isinstance(file_source, (bytes, io.BytesIO)):
            stream = io.BytesIO(file_source) if isinstance(file_source, bytes) else file_source
            doc = docx.Document(stream)
        else:
            raise ValueError(f"Unsupported file source type: {type(file_source)}")

        pages: List[ExtractedPage] = []
        current_page_text: List[str] = []
        current_page_num = 1
        page_word_count = 0
        total_chars = 0
        total_words = 0
        paragraph_index = 0

        # Heuristic: standard printed page contains ~400 words
        WORDS_PER_PAGE_ESTIMATE = 400

        for para in doc.paragraphs:
            text = clean_text(para.text)
            if not text:
                continue

            words = text.split()
            w_count = len(words)
            paragraph_index += 1

            # Check if paragraph contains explicit XML page break: w:br[w:type="page"]
            has_explicit_break = any(
                'type="page"' in run._element.xml
                for run in para.runs
                if hasattr(run, "_element")
            )

            current_page_text.append(text)
            page_word_count += w_count
            total_words += w_count
            total_chars += len(text)

            # Advance page if explicit page break found or estimated page word count exceeded
            if has_explicit_break or page_word_count >= WORDS_PER_PAGE_ESTIMATE:
                page_content = "\n\n".join(current_page_text)
                pages.append(
                    ExtractedPage(
                        page_number=current_page_num,
                        text=page_content,
                        metadata={
                            "page_number": current_page_num,
                            "estimated_page": True,
                            "word_count": len(page_content.split()),
                            "char_count": len(page_content),
                        }
                    )
                )
                current_page_num += 1
                current_page_text = []
                page_word_count = 0

        # Flush any remaining text in the last page buffer
        if current_page_text:
            page_content = "\n\n".join(current_page_text)
            pages.append(
                ExtractedPage(
                    page_number=current_page_num,
                    text=page_content,
                    metadata={
                        "page_number": current_page_num,
                        "estimated_page": True,
                        "word_count": len(page_content.split()),
                        "char_count": len(page_content),
                    }
                )
            )

        # Fallback if document has only tables or no paragraphs
        if not pages and doc.tables:
            table_text = []
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [clean_text(cell.text) for cell in row.cells if clean_text(cell.text)]
                    if row_cells:
                        table_text.append(" | ".join(row_cells))
            full_table_text = "\n".join(table_text)
            if full_table_text:
                pages.append(
                    ExtractedPage(
                        page_number=1,
                        text=full_table_text,
                        metadata={"page_number": 1, "is_table_content": True}
                    )
                )

        return ExtractedDocument(
            filename=filename,
            file_type="docx",
            total_pages=len(pages),
            pages=pages,
            total_characters=total_chars,
            total_words=total_words,
        )

    @classmethod
    def extract(cls, file_source: Union[str, Path, bytes, io.BytesIO], filename: str) -> ExtractedDocument:
        """
        Dispatches extraction based on file extension.
        """
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext == "pdf":
            return cls.extract_pdf(file_source, filename)
        elif ext == "docx":
            return cls.extract_docx(file_source, filename)
        else:
            raise ValueError(f"Unsupported file format '.{ext}'. Only PDF and DOCX are supported.")
