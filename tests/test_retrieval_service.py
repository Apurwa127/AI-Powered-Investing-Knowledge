
from unittest.mock import MagicMock

import pytest

from app.retrieval_service import RetrievalService
from app.embedding_service import EmbeddingService


def test_retrieve_documents():

    # Mock the embedding service
    embedding_service = MagicMock()

    embedding_service.generate_embedding.return_value = [0.1] * 1536

    # Create the retrieval service
    retrieval_service = RetrievalService(embedding_service)

    # Mock Azure AI Search response
    retrieval_service.search_client.search = MagicMock(
        return_value=[
            {
                "content": "Apple reported annual revenue of $391 billion.",
                "company": "Apple",
                "fiscal_year": 2024,
                "source": "2024_Apple.pdf",
                "page_number": 10,
                "document_id": "apple_2024",
                "@search.score": 0.95
            }
        ]
    )

    # Execute retrieval
    results = retrieval_service.retrieve_documents(
        question="What was Apple's revenue in 2024?",
        top_k=3,
        company="Apple",
        fiscal_year=2024
    )

    # Verify results
    assert len(results) == 1
    assert results[0]["company"] == "Apple"
    assert results[0]["fiscal_year"] == 2024
    assert results[0]["score"] == 0.95

    # Verify embedding generation
    embedding_service.generate_embedding.assert_called_once_with(
        "What was Apple's revenue in 2024?"
    )

    # Verify Azure Search filters
    search_call = retrieval_service.search_client.search.call_args

    assert search_call.kwargs["filter"] == (
        "company eq 'Apple' and fiscal_year eq 2024"
    )


def test_retrieve_documents_company_filter():

    embedding_service = MagicMock()
    embedding_service.generate_embedding.return_value = [0.1] * 1536

    retrieval_service = RetrievalService(embedding_service)

    retrieval_service.search_client.search = MagicMock(return_value=[])

    retrieval_service.retrieve_documents(
        question="What was Apple's revenue?",
        company="Apple"
    )

    search_call = retrieval_service.search_client.search.call_args

    assert search_call.kwargs["filter"] == "company eq 'Apple'"


def test_retrieve_documents_fiscal_year_filter():

    embedding_service = MagicMock()
    embedding_service.generate_embedding.return_value = [0.1] * 1536

    retrieval_service = RetrievalService(embedding_service)

    retrieval_service.search_client.search = MagicMock(return_value=[])

    retrieval_service.retrieve_documents(
        question="What was the company's revenue?",
        fiscal_year=2024
    )

    search_call = retrieval_service.search_client.search.call_args

    assert search_call.kwargs["filter"] == "fiscal_year eq 2024"


def test_retrieve_documents_without_filters():

    embedding_service = MagicMock()
    embedding_service.generate_embedding.return_value = [0.1] * 1536

    retrieval_service = RetrievalService(embedding_service)

    retrieval_service.search_client.search = MagicMock(return_value=[])

    retrieval_service.retrieve_documents(
        question="What was the company's revenue?"
    )

    search_call = retrieval_service.search_client.search.call_args

    assert search_call.kwargs["filter"] is None


def test_retrieve_documents_empty_question():

    embedding_service = MagicMock()
    retrieval_service = RetrievalService(embedding_service)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieval_service.retrieve_documents("")


def test_retrieve_documents_invalid_top_k():

    embedding_service = MagicMock()
    retrieval_service = RetrievalService(embedding_service)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retrieval_service.retrieve_documents(
            "What was Apple's revenue?",
            top_k=0
        )


def test_live_retrieval():

    embedding_service = EmbeddingService()

    retrieval_service = RetrievalService(embedding_service)

    results = retrieval_service.retrieve_documents(
        question="What were Apple's major sources of revenue in 2024?",
        top_k=3,
        company="Apple",
        fiscal_year=2024
    )

    print("\nRetrieved Apple 2024 documents:")

    for document in results:
        print("\nCompany:", document["company"])
        print("Page:", document["page_number"])
        print("Score:", document["score"])
        print("Content:", document["content"][:500])

    assert len(results) > 0
