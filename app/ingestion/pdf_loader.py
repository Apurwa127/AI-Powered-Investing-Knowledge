
from pathlib import Path
import logging

import pymupdf
import pymupdf4llm

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Extract page-level Markdown from PDF documents.

    Supports:
    1. Existing PDFs stored on disk.
    2. PDFs uploaded by users as bytes.
    """

    def __init__(self, max_file_size_mb: int = 100):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def load_from_path(
        self,
        pdf_path: str | Path,
        document_id: str,
        company: str,
        fiscal_year: int,
    ) -> list[Document]:

        pdf_file = Path(pdf_path)

        if not pdf_file.is_file():
            raise FileNotFoundError(
                f"PDF not found: {pdf_file}"
            )

        file_size = pdf_file.stat().st_size

        if file_size == 0:
            raise ValueError("PDF file is empty")

        if file_size > self.max_file_size_bytes:
            raise ValueError("PDF exceeds maximum file size")

        logger.info("Extracting PDF: %s", pdf_file.name)

        with pymupdf.open(str(pdf_file)) as pdf:
            pages = pymupdf4llm.to_markdown(
                pdf,
                page_chunks=True,
                use_ocr=False
            )

        return self._create_documents(
            pages=pages,
            document_id=document_id,
            company=company,
            fiscal_year=fiscal_year,
            source=pdf_file.name,
        )

    def load_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        document_id: str,
        company: str,
        fiscal_year: int,
    ) -> list[Document]:

        if not file_bytes:
            raise ValueError("Uploaded PDF is empty")

        if len(file_bytes) > self.max_file_size_bytes:
            raise ValueError("PDF exceeds maximum file size")

        if not file_bytes.startswith(b"%PDF-"):
            raise ValueError("Uploaded file is not a valid PDF")

        logger.info("Processing uploaded PDF: %s", filename)

        with pymupdf.open(
            stream=file_bytes,
            filetype="pdf"
        ) as pdf:

            pages = pymupdf4llm.to_markdown(
                pdf,
                page_chunks=True,
                use_ocr=False
            )

        return self._create_documents(
            pages=pages,
            document_id=document_id,
            company=company,
            fiscal_year=fiscal_year,
            source=filename,
        )

    @staticmethod
    def _create_documents(
        pages: list[dict],
        document_id: str,
        company: str,
        fiscal_year: int,
        source: str,
    ) -> list[Document]:

        documents = []

        if not pages:
            raise ValueError(
                "PDF extraction returned no pages"
            )

        for page in pages:

            page_text = page.get("text", "").strip()
            page_metadata = page.get("metadata", {})

            page_number = page_metadata.get("page_number")

            if not page_text:
                logger.warning(
                    "Page %s contains no extractable text",
                    page_number
                )
                continue

            metadata = {
                "document_id": document_id,
                "company": company,
                "fiscal_year": fiscal_year,
                "document_type": "annual_report",
                "source": source,
                "page_number": page_number,
                "page_count": page_metadata.get("page_count"),
                "table_count": len(page.get("tables", [])),
            }

            documents.append(
                Document(
                    page_content=page_text,
                    metadata=metadata
                )
            )

        if not documents:
            raise ValueError(
                "No readable text was extracted from the PDF"
            )

        logger.info(
            "Extracted %d page documents from %s",
            len(documents),
            source
        )

        return documents
