
import os

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from app.embedding_service import EmbeddingService


load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = "investing-knowledge-index"


class RetrievalService:

    def __init__(self, embedding_service: EmbeddingService):

        if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_API_KEY:
            raise ValueError("Azure AI Search credentials are missing")

        self.embedding_service = embedding_service

        self.search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
        )

    def retrieve_documents(
        self,
        question: str,
        top_k: int = 3,
        company: str | None = None,
        fiscal_year: int | None = None
    ) -> list[dict]:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Generate embedding for the user's question
        query_embedding = self.embedding_service.generate_embedding(question)

        # Configure vector search
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )

        # Build optional metadata filters
        filters = []

        if company:
            safe_company = company.replace("'", "''")
            filters.append(f"company eq '{safe_company}'")

        if fiscal_year is not None:
            filters.append(f"fiscal_year eq {fiscal_year}")

        search_filter = " and ".join(filters) if filters else None

        # Search Azure AI Search
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=search_filter,
            select=[
                "content",
                "company",
                "fiscal_year",
                "source",
                "page_number",
                "document_id"
            ],
            top=top_k
        )

        documents = []

        for result in results:
            documents.append({
                "content": result["content"],
                "company": result.get("company"),
                "fiscal_year": result.get("fiscal_year"),
                "source": result.get("source"),
                "page_number": result.get("page_number"),
                "document_id": result.get("document_id"),
                "score": result.get("@search.score")
            })

        return documents
