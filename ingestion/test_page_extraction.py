
from pathlib import Path
import pymupdf4llm

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent

# Locate the PDF
pdf_path = project_root / "data" / "2024_Apple.pdf"

# Validate the path
if not pdf_path.is_file():
    raise FileNotFoundError(
        f"PDF not found: {pdf_path}"
    )

print(f"Processing: {pdf_path}")

# Extract one dictionary per page
pages = pymupdf4llm.to_markdown(
    str(pdf_path),
    page_chunks=True
)

print(f"Total pages: {len(pages)}")

for page in pages[:3]:

    metadata = page["metadata"]

    print("\n-------------------------")
    print(f"Page: {metadata.get('page_number')}")
    print(f"Title: {metadata.get('title')}")

    print("Content preview:")
    print(page["text"][:500])

    print(
        "Tables detected:",
        len(page.get("tables", []))
    )
