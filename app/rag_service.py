
import re

from app.retrieval_service import RetrievalService
from app.llm_service import LLMService
from app.reranker import Reranker


class RAGService:

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        reranker: Reranker
    ):

        self.retrieval_service = retrieval_service
        self.llm_service = llm_service
        self.reranker = reranker

    def ask_question(
        self,
        question: str,
        company: str | None = None,
        fiscal_year: int | None = None,
        top_k: int = 3,
        retrieval_k: int = 10
    ) -> dict:

        # Validate inputs
        if not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than zero")

        if retrieval_k < top_k:
            raise ValueError(
                "retrieval_k must be greater than or equal to top_k"
            )

        # Step 1: Retrieve candidate documents
        retrieved_documents = self.retrieval_service.retrieve_documents(
            question=question,
            top_k=retrieval_k,
            company=company,
            fiscal_year=fiscal_year,
            search_mode="hybrid"
        )

        if not retrieved_documents:
            return {
                "question": question,
                "answer": "No relevant information was found in the available documents.",
                "sources": [],
                "invalid_citations": []
            }

        # Step 2: Rerank retrieved documents
        reranked_documents = self.reranker.rerank(
            question=question,
            documents=retrieved_documents,
            top_k=top_k
        )

        if not reranked_documents:
            return {
                "question": question,
                "answer": "No relevant information was found after reranking.",
                "sources": [],
                "invalid_citations": []
            }

        # Step 3: Combine reranked content into context
        context = "\n\n".join(
            f"""
            Source: {document['source']}
            Company: {document['company']}
            Fiscal Year: {document['fiscal_year']}
            Page Number: {document['page_number']}

            Content:
            {document['content']}
            """
            for document in reranked_documents
        )

        # Step 4: Generate an answer using the LLM
        answer = self.llm_service.generate_answer(
            question=question,
            context=context
        )

        # Step 5: Validate citations
        invalid_citations = self.validate_sources(
            answer=answer,
            retrieved_documents=reranked_documents
        )

        # Step 6: Extract source information
        sources = [
            {
                "company": document["company"],
                "fiscal_year": document["fiscal_year"],
                "source": document["source"],
                "page_number": document["page_number"],
                "rerank_score": document["rerank_score"]
            }
            for document in reranked_documents
        ]

        # Step 7: Return final result
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
