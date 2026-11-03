from unittest.mock import MagicMock

import pytest

from app.rag_service import RAGService
from app.embedding_service import EmbeddingService
from app.retrieval_service import RetrievalService
from app.llm_service import LLMService


@pytest.fixture
def rag_service():

    retrieval_service = MagicMock()
    llm_service = MagicMock()

    return RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service
    )


def test_ask_question_success(rag_service):

    rag_service.retrieval_service.retrieve_documents.return_value = [
        {
            "content": "Apple reported net sales of $391 billion.",
            "company": "Apple",
            "fiscal_year": 2024,
            "source": "2024_Apple.pdf",
            "page_number": 15,
            "document_id": "apple_2024_annual",
            "score": 0.85
        }
    ]

    rag_service.llm_service.generate_answer.return_value = (
        "Apple reported net sales of $391 billion in 2024."
    )

    result = rag_service.ask_question(
        question="What was Apple's revenue in 2024?",
        company="Apple",
        fiscal_year=2024
    )

    assert result["answer"] == (
        "Apple reported net sales of $391 billion in 2024."
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["company"] == "Apple"
    assert result["sources"][0]["page_number"] == 15

    rag_service.retrieval_service.retrieve_documents.assert_called_once()
    rag_service.llm_service.generate_answer.assert_called_once()


def test_ask_question_no_documents(rag_service):

    rag_service.retrieval_service.retrieve_documents.return_value = []

    result = rag_service.ask_question(
        question="What was Apple's revenue?"
    )

    assert result["sources"] == []

    rag_service.llm_service.generate_answer.assert_not_called()


def test_ask_question_empty_question(rag_service):

    with pytest.raises(ValueError, match="Question cannot be empty"):
        rag_service.ask_question(question=" ")


def test_ask_question_invalid_top_k(rag_service):

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        rag_service.ask_question(
            question="What was Apple's revenue?",
            top_k=0
        )

@pytest.mark.integration
def test_rag_workflow_real_services():

    embedding_service = EmbeddingService()

    retrieval_service = RetrievalService(
        embedding_service=embedding_service
    )

    llm_service = LLMService()

    rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_service=llm_service
    )

    result = rag_service.ask_question(
        question="What were Apple's major sources of revenue in 2024?",
        company="Apple",
        fiscal_year=2024,
        top_k=3
    )

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(source)

    assert result["answer"]
    assert len(result["sources"]) > 0
    assert all(source["company"] == "Apple" for source in result["sources"])

def test_validate_sources_all_valid(rag_service):

    documents = [
        {
            "source": "2024_Apple.pdf",
            "page_number": 10
        },
        {
            "source": "2024_Apple.pdf",
            "page_number": 34
        }
    ]

    answer = (
        "Apple reported revenue of $391 billion. "
        "[Source: 2024_Apple.pdf, Page 10]"
    )

    invalid = rag_service.validate_sources(
        answer=answer,
        retrieved_documents=documents
    )

    assert invalid == []


def test_validate_sources_invalid_citation(rag_service):

    documents = [
        {
            "source": "2024_Apple.pdf",
            "page_number": 10
        }
    ]

    answer = (
        "Apple reported revenue of $391 billion. "
        "[Source: 2024_Apple.pdf, Page 50]"
    )

    invalid = rag_service.validate_sources(
        answer=answer,
        retrieved_documents=documents
    )

    assert invalid == ["2024_Apple.pdf, Page 50"]