
from pathlib import Path
import logging
import tempfile
import os

import pymupdf4llm


logger = logging.getLogger(__name__)


class PDFToMarkdownConverter:
    """
    Convert PDF documents to Markdown with basic
    production-oriented validation and error handling.
    """

    def __init__(
        self,
        max_file_size_mb: int = 100,
        overwrite: bool = True
    ):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.overwrite = overwrite

    def convert_pdf(
        self,
        pdf_path: str,
        output_dir: str
    ) -> str:

        pdf_file = Path(pdf_path)

        # 1. Validate input file
        if not pdf_file.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_file.suffix.lower() != ".pdf":
            raise ValueError(
                f"Unsupported file type: {pdf_file.suffix}"
            )

        # 2. Validate file size
        file_size = pdf_file.stat().st_size

        if file_size == 0:
            raise ValueError("PDF file is empty")

        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"PDF exceeds maximum allowed size: {pdf_file.name}"
            )

        output_path = Path(output_dir)
        output_path.mkdir(
            parents=True,
            exist_ok=True
        )

        markdown_file = output_path / f"{pdf_file.stem}.md"

        if markdown_file.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output already exists: {markdown_file}"
            )

        logger.info(
            "Starting PDF conversion: %s",
            pdf_file.name
        )

        try:

            # 3. Extract PDF content
            markdown_content = pymupdf4llm.to_markdown(
                str(pdf_file)
            )

            # 4. Validate extracted content
            if not markdown_content or not markdown_content.strip():
                raise ValueError(
                    f"No extractable content found: {pdf_file.name}"
                )

            # 5. Write to a temporary file first
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".tmp",
                    dir=output_path,
                    delete=False
                ) as temp_file:

                    temp_file.write(markdown_content)
                    temp_path = Path(temp_file.name)

                # 6. Atomically replace destination
                os.replace(
                    temp_path,
                    markdown_file
                )

            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink()

            logger.info(
                "Successfully converted: %s",
                markdown_file.name
            )

            return str(markdown_file)

        except Exception:
            logger.exception(
                "PDF conversion failed: %s",
                pdf_file.name
            )
            raise

    def convert_directory(
        self,
        input_dir: str,
        output_dir: str,
        continue_on_error: bool = True
    ) -> list[str]:

        input_path = Path(input_dir)

        if not input_path.is_dir():
            raise NotADirectoryError(
                f"Input directory not found: {input_dir}"
            )

        # Include .pdf and .PDF files
        pdf_files = sorted(
            file for file in input_path.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".pdf"
        )

        if not pdf_files:
            logger.warning(
                "No PDF documents found in %s",
                input_dir
            )

        markdown_files = []
        failed_files = []

        for pdf_file in pdf_files:

            try:
                markdown_file = self.convert_pdf(
                    pdf_path=str(pdf_file),
                    output_dir=output_dir
                )

                markdown_files.append(markdown_file)

            except Exception:

                failed_files.append(pdf_file.name)

                if not continue_on_error:
                    raise

        logger.info(
            "Conversion summary: %d successful, %d failed",
            len(markdown_files),
            len(failed_files)
        )

        if failed_files:
            logger.error(
                "Failed documents: %s",
                failed_files
            )

        return markdown_files


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    repo_root = Path(__file__).resolve().parent.parent

    input_dir = repo_root / "data"
    output_dir = repo_root / "data" / "markdown"

    converter = PDFToMarkdownConverter(
        max_file_size_mb=100,
        overwrite=True
    )

    markdown_files = converter.convert_directory(
        input_dir=str(input_dir),
        output_dir=str(output_dir)
    )

    logger.info(
        "Successfully converted %d documents",
        len(markdown_files)
    )
