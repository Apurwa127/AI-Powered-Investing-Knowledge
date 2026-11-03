
from app.ingestion.ingestion_pipeline import IngestionPipeline


def main():

    # Initialize ingestion pipeline
    pipeline = IngestionPipeline()

    # Ingest Apple's 2024 annual report
    result = pipeline.ingest_pdf(
        pdf_path="data/2024_Apple.pdf",
        document_id="apple_2024_annual",
        company="Apple",
        fiscal_year=2024
    )

    # Display ingestion summary
    print("\nIngestion completed!")
    print(result)


if __name__ == "__main__":
    main()
