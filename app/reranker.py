
from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        question: str,
        documents: list[dict],
        top_k: int = 3
    ) -> list[dict]:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not documents:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        pairs = [
            [question, document["content"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        for document, score in zip(documents, scores):
            document["rerank_score"] = float(score)

        reranked_documents = sorted(
            documents,
            key=lambda document: document["rerank_score"],
            reverse=True
        )

        return reranked_documents[:top_k]
