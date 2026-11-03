import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMService:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key is missing")

        self.client = OpenAI(api_key=api_key)

        self.model = "gpt-5.4-mini"

    def generate_answer(
        self,
        question: str,
        context: str
    ) -> str:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not context.strip():
            raise ValueError("Context cannot be empty")

        response = self.client.responses.create(
            model=self.model,
            instructions=(
                 "You are a financial research assistant. "
                 "Answer the user's question using only the "
                 "provided financial report context. "
                 "Do not invent financial figures or unsupported claims. "
                 "If the answer is not available in the context, "
                 "say that the available documents do not provide "
                 "enough information. "
                 "Cite every important financial claim using the "
                 "exact source filename and page number provided "
                 "in the context. "
                 "Use this citation format: "
                 "[Source: document_name, Page X]. "
                 "Do not cite pages that were not provided in the context. "
                 "If multiple passages support different claims, "
                 "cite the relevant passage for each claim. "
                 "Keep the answer concise but complete."
            ),
            input=f"""
            Financial Report Context:
            {context}

            User Question:
            {question}
            """,
            max_output_tokens=300
        )

        return response.output_text