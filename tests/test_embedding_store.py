
from langchain_core.documents import Document

from app.embedding_service import EmbeddingService
from app.embedding_store import EmbeddingStore


def test_generate_chunk_embeddings():

    embedding_service = EmbeddingService()

    embedding_store = EmbeddingStore(
        embedding_service=embedding_service
    )

    documents = [
        Document(
            page_content="Apple reported strong revenue growth in 2024.",
            metadata={
                "document_id": "apple_2024_annual",
                "company": "Apple",
                "fiscal_year": 2024,
                "document_type": "annual_report",
                "source": "2024_Apple.pdf",
                "page_number": 10,
                "chunk_index": 0
            }
        ),
        Document(
            page_content="Microsoft expanded its cloud computing business.",
            metadata={
                "document_id": "microsoft_2024_annual",
                "company": "Microsoft",
                "fiscal_year": 2024,
                "document_type": "annual_report",
                "source": "2024_Microsoft.pdf",
                "page_number": 15,
                "chunk_index": 1
            }
        )
    ]

    records = embedding_store.generate_chunk_embeddings(documents)

    assert len(records) == 2

    for record in records:
        assert record["id"]
        assert record["content"]
        assert len(record["content_vector"]) == 1536
        assert record["source"]
        assert isinstance(record["chunk_index"], int)
        assert record["document_id"]
        assert record["page_number"] is not None
def test_upload_to_azure_search():

    embedding_service = EmbeddingService()

    embedding_store = EmbeddingStore(
        embedding_service=embedding_service
    )

    documents = [
        Document(
            page_content="This is a test document for Azure AI Search.",
            metadata={
                "document_id": "test_document",
                "company": "Apple",
                "fiscal_year": 2024,
                "document_type": "annual_report",
                "source": "test_document.pdf",
                "page_number": 1,
                "chunk_index": 0
            }
        )
    ]

    records = embedding_store.generate_chunk_embeddings(documents)

    uploaded_count = embedding_store.upload_to_azure_search(records)

    assert uploaded_count == 1

def test_generate_chunk_id_is_deterministic():

    first_id = EmbeddingStore.generate_chunk_id(
        document_id="apple_2024_annual",
        page_number=10,
        chunk_index=2
    )

    second_id = EmbeddingStore.generate_chunk_id(
        document_id="apple_2024_annual",
        page_number=10,
        chunk_index=2
    )

    assert first_id == second_id


def test_generate_chunk_id_is_unique():

    first_id = EmbeddingStore.generate_chunk_id(
        document_id="apple_2024_annual",
        page_number=10,
        chunk_index=2
    )

    second_id = EmbeddingStore.generate_chunk_id(
        document_id="apple_2024_annual",
        page_number=10,
        chunk_index=3
    )

    assert first_id != second_id