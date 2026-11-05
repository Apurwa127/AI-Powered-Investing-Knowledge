
from app.embedding_service import EmbeddingService
from app.retrieval_service import RetrievalService

from evaluation.evaluation_dataset import evaluation_dataset
from evaluation.retrieval_metrics import evaluate_retrieval


def main():

    # Initialize services
    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService(embedding_service)

    k = 3
    all_results = []

    for item in evaluation_dataset:

        print("\n" + "=" * 60)
        print("Question:", item["question"])
        print("Company:", item["company"])

        # Retrieve documents
        documents = retrieval_service.retrieve_documents(
            question=item["question"],
            top_k=k,
            company=item["company"],
            fiscal_year=item["fiscal_year"]
        )
        # Inspect retrieved documents
        for document in documents:
            print("ID:", document["id"])
            print("Page:", document["page_number"])
            print("Score:", document["score"])
            print("Content:", document["content"][:500])
            print("-" * 60)

        # Extract retrieved chunk IDs
        retrieved_ids = [
            document["id"]
            for document in documents
        ]

        # Get ground-truth chunk IDs
        relevant_ids = item["relevant_chunk_ids"]

        # Calculate metrics
        metrics = evaluate_retrieval(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k
        )

        all_results.append(metrics)

        print("Retrieved chunks:", len(retrieved_ids))
        print("Precision@3:", round(metrics["precision_at_k"], 3))
        print("Recall@3:", round(metrics["recall_at_k"], 3))
        print("F1@3:", round(metrics["f1_at_k"], 3))

    # Calculate average metrics
    print("\n" + "=" * 60)
    print("OVERALL RETRIEVAL EVALUATION")
    print("=" * 60)

    for metric in [
        "precision_at_k",
        "recall_at_k",
        "f1_at_k"
    ]:

        average = sum(
            result[metric]
            for result in all_results
        ) / len(all_results)

        print(f"Average {metric}: {average:.3f}")


if __name__ == "__main__":
    main()
