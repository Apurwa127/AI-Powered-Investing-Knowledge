
from pathlib import Path

from app.ingestion.pdf_loader import PDFLoader


def test_apple_pdf_extraction():

    # Locate project root
    project_root = Path(__file__).resolve().parents[1]

    # Locate Apple annual report
    pdf_path = project_root / "data" / "2024_Apple.pdf"

    # Initialize PDF loader
    loader = PDFLoader()

    # Extract PDF pages
    pages = loader.load_from_path(
        pdf_path=pdf_path,
        document_id="apple_2024_annual",
        company="Apple",
        fiscal_year=2024
    )

    # Validate extraction
    assert len(pages) > 0, "No pages extracted"

    # Validate page metadata
    for page in pages:

        assert page.page_content.strip()

        assert (
            page.metadata["document_id"]
            == "apple_2024_annual"
        )

        assert page.metadata["company"] == "Apple"

        assert page.metadata["fiscal_year"] == 2024

        assert page.metadata["page_number"] is not None

    # Display results
    print("\nAll validation checks passed!")

    print("Total pages extracted:", len(pages))

    print(
        "First page:",
        pages[0].metadata["page_number"]
    )

    print(
        "Last page:",
        pages[-1].metadata["page_number"]
    )
