from app.retrieval_service import RetrievalService
from app.llm_service import LLMService
import re

from app.retrieval_service import RetrievalService
from app.llm_service import LLMService


class RAGService:

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService
    ):

        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    def ask_question(
            self,
            question: str,
            company: str | None = None,
            fiscal_year: int | None = None,
            top_k: int = 3
    ) -> dict:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Step 1: Retrieve relevant documents
        retrieved_documents = self.retrieval_service.retrieve_documents(
            question=question,
            top_k=top_k,
            company=company,
            fiscal_year=fiscal_year
        )

        if not retrieved_documents:
            return {
                "question": question,
                "answer": "No relevant information was found in the available documents.",
                "sources": []
            }

        # Step 2: Combine retrieved content into context
        context = "\n\n".join(
            f"""
            Source: {document['source']}
            Company: {document['company']}
            Fiscal Year: {document['fiscal_year']}
            Page Number: {document['page_number']}

            Content:
            {document['content']}
            """
            for document in retrieved_documents
        )

        # Step 3: Generate an answer using the LLM
        answer = self.llm_service.generate_answer(
            question=question,
            context=context
        )
        invalid_citations = self.validate_sources(
            answer=answer,
            retrieved_documents=retrieved_documents
        )

        # Step 4: Extract source information
        sources = [
            {
                "company": document["company"],
                "fiscal_year": document["fiscal_year"],
                "source": document["source"],
                "page_number": document["page_number"]
            }
            for document in retrieved_documents
        ]

        # Step 5: Return the final result
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "invalid_citations": invalid_citations
        }

    def validate_sources(
            self,
            answer: str,
            retrieved_documents: list[dict]
    ) -> list[str]:

        valid_sources = {
            (
                document["source"],
                document["page_number"]
            )
            for document in retrieved_documents
        }

        citations = re.findall(
            r"\[Source: (.*?), Page (\d+)\]",
            answer
        )

        invalid_citations = []

        for source, page_number in citations:

            citation = (source, int(page_number))

            if citation not in valid_sources:
                invalid_citations.append(
                    f"{source}, Page {page_number}"
                )

        return invalid_citations