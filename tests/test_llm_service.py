from app.llm_service import LLMService
from unittest.mock import MagicMock, patch

from app.llm_service import LLMService
import pytest


def test_llm_service_initialization():

    llm_service = LLMService()

    assert llm_service.client is not None
    assert llm_service.model == "gpt-5.4-mini"

def test_llm_service_initialization():

    llm_service = LLMService()

    assert llm_service.client is not None
    assert llm_service.model == "gpt-5.4-mini"


@patch("app.llm_service.OpenAI")
def test_generate_answer(mock_openai):

    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_response = MagicMock()
    mock_response.output_text = (
        "Apple's major revenue sources included iPhone and Services."
    )

    mock_client.responses.create.return_value = mock_response

    llm_service = LLMService()

    answer = llm_service.generate_answer(
        question="What were Apple's major revenue sources?",
        context=(
            "Apple's 2024 annual report identifies "
            "iPhone and Services as major revenue categories."
        )
    )

    assert answer == (
        "Apple's major revenue sources included iPhone and Services."
    )

    mock_client.responses.create.assert_called_once()

@pytest.mark.integration
def test_generate_answer_real_api():

    llm_service = LLMService()

    context = """
    Apple's 2024 annual report reported total net sales
    of $391.0 billion.
    """

    answer = llm_service.generate_answer(
        question="What were Apple's total net sales in 2024?",
        context=context
    )

    print("\nGenerated answer:", answer)

    assert answer
    assert isinstance(answer, str)
    assert "391" in answer