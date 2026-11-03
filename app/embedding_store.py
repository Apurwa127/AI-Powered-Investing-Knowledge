
import hashlib
import logging
import os

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_core.documents import Document

from app.embedding_service import EmbeddingService


load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = "investing-knowledge-index"

logger = logging.getLogger(__name__)


class EmbeddingStore:

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    @staticmethod
    def generate_chunk_id(
        document_id: str,
        page_number: int,
        chunk_index: int
    ) -> str:

        unique_key = (
            f"{document_id}_{page_number}_{chunk_index}"
        )

        return hashlib.sha256(
            unique_key.encode("utf-8")
        ).hexdigest()

    def generate_chunk_embeddings(
        self,
        documents: list[Document]
    ) -> list[dict]:

        if not documents:
            raise ValueError("No documents supplied for embedding")

        texts = [
            document.page_content
            for document in documents
        ]

        vectors = self.embedding_service.generate_embeddings(texts)

        if len(vectors) != len(documents):
            raise ValueError("Mismatch between documents and embeddings")

        records = []

        for document, vector in zip(documents, vectors):

            if len(vector) != 1536:
                raise ValueError("Invalid embedding dimensions")

            metadata = document.metadata

            document_id = metadata.get("document_id")
            page_number = metadata.get("page_number")
            chunk_index = metadata.get("chunk_index", 0)

            if document_id is None or page_number is None:
                raise ValueError(
                    "Document ID and page number are required"
                )

            record = {
                "id": self.generate_chunk_id(
                    document_id=document_id,
                    page_number=page_number,
                    chunk_index=chunk_index
                ),

                "content": document.page_content,
                "content_vector": vector,

                # Document metadata
                "document_id": document_id,
                "company": metadata.get("company"),
                "fiscal_year": metadata.get("fiscal_year"),
                "document_type": metadata.get("document_type"),
                "source": metadata.get("source", "unknown"),

                # Chunk metadata
                "page_number": page_number,
                "chunk_index": chunk_index,
            }

            records.append(record)

        logger.info(
            "Generated embeddings for %d chunks",
            len(records)
        )

        return records

    def upload_to_azure_search(self, records: list[dict]):

        if not records:
            raise ValueError(
                "No records supplied for Azure AI Search upload"
            )

        if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
            raise ValueError(
                "Azure AI Search credentials are missing"
            )

        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
        )

        results = search_client.upload_documents(
            documents=records
        )

        if len(results) != len(records):
            raise RuntimeError(
                "Mismatch between uploaded records and results"
            )

        successful = sum(
            1 for result in results if result.succeeded
        )

        failed = len(results) - successful

        logger.info(
            "Uploaded %d records successfully; %d failed",
            successful,
            failed
        )

        if failed:
            errors = [
                result.error_message
                for result in results
                if not result.succeeded
            ]

            raise RuntimeError(
                f"Azure AI Search upload failed: {errors}"
            )

        return successful
