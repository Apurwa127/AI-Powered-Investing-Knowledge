
from app.ingestion.ingestion_pipeline import IngestionPipeline


def main():

    # Initialize ingestion pipeline once
    pipeline = IngestionPipeline()

    # Define the annual reports
    reports = [
        {
            "pdf_path": "data/2024_Apple.pdf",
            "document_id": "apple_2024_annual",
            "company": "Apple",
            "fiscal_year": 2024
        },
        {
            "pdf_path": "data/2024_Microsoft.pdf",
            "document_id": "microsoft_2024_annual",
            "company": "Microsoft",
            "fiscal_year": 2024
        },
        {
            "pdf_path": "data/2024_Tesla.pdf",
            "document_id": "tesla_2024_annual",
            "company": "Tesla",
            "fiscal_year": 2024
        }
    ]

    # Ingest all reports
    for report in reports:

        print(f"\nIngesting {report['company']} 2024 annual report...")

        result = pipeline.ingest_pdf(**report)

        print("Ingestion completed!")
        print(result)


if __name__ == "__main__":
    main()
