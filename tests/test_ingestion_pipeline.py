from unittest.mock import MagicMock, patch

from app.ingestion.ingestion_pipeline import IngestionPipeline

def test_ingestion_pipeline_initialization():

    with patch("app.ingestion.ingestion_pipeline.EmbeddingService") as mock_embedding, \
         patch("app.ingestion.ingestion_pipeline.PDFLoader") as mock_pdf_loader, \
         patch("app.ingestion.ingestion_pipeline.SemanticChunkingService") as mock_chunker, \
         patch("app.ingestion.ingestion_pipeline.EmbeddingStore") as mock_store:

        mock_embedding.return_value.embeddings = MagicMock()

        pipeline = IngestionPipeline()

        assert pipeline.embedding_service is not None
        assert pipeline.pdf_loader is not None
        assert pipeline.chunker is not None
        assert pipeline.embedding_store is not None

def test_process_pdf():

    with patch("app.ingestion.ingestion_pipeline.EmbeddingService"), \
         patch("app.ingestion.ingestion_pipeline.PDFLoader") as mock_pdf_loader, \
         patch("app.ingestion.ingestion_pipeline.SemanticChunkingService") as mock_chunker, \
         patch("app.ingestion.ingestion_pipeline.EmbeddingStore"):

        pipeline = IngestionPipeline()

        # Simulate pages extracted from a PDF
        mock_pages = [MagicMock(), MagicMock()]

        mock_pdf_loader.return_value.load_from_path.return_value = mock_pages

        # Simulate semantic chunks
        mock_chunks = [MagicMock(), MagicMock(), MagicMock()]

        mock_chunker.return_value.chunk_documents.return_value = mock_chunks

        # Execute the method
        chunks = pipeline.process_pdf(
            pdf_path="data/2024_Apple.pdf",
            document_id="apple_2024_annual",
            company="Apple",
            fiscal_year=2024
        )

        # Verify the results
        assert chunks == mock_chunks

        mock_pdf_loader.return_value.load_from_path.assert_called_once_with(
            pdf_path="data/2024_Apple.pdf",
            document_id="apple_2024_annual",
            company="Apple",
            fiscal_year=2024
        )

        mock_chunker.return_value.chunk_documents.assert_called_once_with(
            mock_pages
        )
def test_ingest_pdf():

    with patch("app.ingestion.ingestion_pipeline.EmbeddingService"), \
         patch("app.ingestion.ingestion_pipeline.PDFLoader") as mock_pdf_loader, \
         patch("app.ingestion.ingestion_pipeline.SemanticChunkingService") as mock_chunker, \
         patch("app.ingestion.ingestion_pipeline.EmbeddingStore") as mock_store:

        pipeline = IngestionPipeline()

        # Simulate two extracted pages
        mock_pages = [MagicMock(), MagicMock()]

        mock_pdf_loader.return_value.load_from_path.return_value = mock_pages

        # Simulate three semantic chunks
        mock_chunks = [
            MagicMock(metadata={"page_number": 1}),
            MagicMock(metadata={"page_number": 1}),
            MagicMock(metadata={"page_number": 2})
        ]

        mock_chunker.return_value.chunk_documents.return_value = mock_chunks

        # Simulate generated embedding records
        mock_records = [
            {"id": "record_1"},
            {"id": "record_2"},
            {"id": "record_3"}
        ]

        mock_store.return_value.generate_chunk_embeddings.return_value = mock_records

        # Simulate successful Azure upload
        mock_store.return_value.upload_to_azure_search.return_value = 3

        # Execute the ingestion pipeline
        result = pipeline.ingest_pdf(
            pdf_path="data/2024_Apple.pdf",
            document_id="apple_2024_annual",
            company="Apple",
            fiscal_year=2024
        )

        # Verify the ingestion summary
        assert result == {
            "company": "Apple",
            "fiscal_year": 2024,
            "pages_processed": 2,
            "chunks_generated": 3,
            "records_uploaded": 3
        }

        # Verify embedding generation was called
        mock_store.return_value.generate_chunk_embeddings.assert_called_once_with(
            mock_chunks
        )

        # Verify Azure upload was called
        mock_store.return_value.upload_to_azure_search.assert_called_once_with(
            mock_records
        )