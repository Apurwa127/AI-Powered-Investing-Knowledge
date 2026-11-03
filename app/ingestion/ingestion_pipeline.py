import logging

from langchain_core.documents import Document

from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.semantic_chunker import SemanticChunkingService
from app.embedding_service import EmbeddingService
from app.embedding_store import EmbeddingStore


logger = logging.getLogger(__name__)


class IngestionPipeline:

    def __init__(self):

        # Initialize embedding service
        self.embedding_service = EmbeddingService()

        # Initialize PDF loader
        self.pdf_loader = PDFLoader()

        # Initialize semantic chunker
        self.chunker = SemanticChunkingService(
            embeddings=self.embedding_service.embeddings
        )

        # Initialize embedding store
        self.embedding_store = EmbeddingStore(
            embedding_service=self.embedding_service
        )

    def process_pdf(
        self,
        pdf_path: str,
        document_id: str,
        company: str,
        fiscal_year: int
    ) -> list[Document]:

        # Step 1: Extract pages from PDF
        pages = self.pdf_loader.load_from_path(
            pdf_path=pdf_path,
            document_id=document_id,
            company=company,
            fiscal_year=fiscal_year
        )

        # Step 2: Convert pages into semantic chunks
        chunks = self.chunker.chunk_documents(pages)

        return chunks

    def ingest_pdf(
        self,
        pdf_path: str,
        document_id: str,
        company: str,
        fiscal_year: int
    ) -> dict:

        # Step 1: Extract PDF pages and generate semantic chunks
        chunks = self.process_pdf(
            pdf_path=pdf_path,
            document_id=document_id,
            company=company,
            fiscal_year=fiscal_year
        )

        # Step 2: Generate embeddings and prepare Azure Search records
        records = self.embedding_store.generate_chunk_embeddings(chunks)

        # Step 3: Upload records to Azure AI Search
        uploaded_count = self.embedding_store.upload_to_azure_search(records)

        # Step 4: Log ingestion summary
        logger.info(
            "Ingestion completed for %s: %d pages, %d chunks, %d uploaded",
            company,
            len(set(
                chunk.metadata["page_number"] for chunk in chunks
            )),
            len(chunks),
            uploaded_count
        )

        # Step 5: Return ingestion summary
        return {
            "company": company,
            "fiscal_year": fiscal_year,
            "pages_processed": len(set(
                chunk.metadata["page_number"] for chunk in chunks
            )),
            "chunks_generated": len(chunks),
            "records_uploaded": uploaded_count
        }