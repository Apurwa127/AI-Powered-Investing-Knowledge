
from app.embedding_service import EmbeddingService
from app.retrieval_service import RetrievalService

from evaluation.evaluation_dataset import evaluation_dataset
from evaluation.retrieval_metrics import evaluate_retrieval


def main():

    # Initialize services
    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService(embedding_service)

    k = 3

    search_modes = [
        "vector",
        "keyword",
        "hybrid"
    ]

    overall_results = {}

    # Run each search strategy
    for search_mode in search_modes:

        print("\n" + "=" * 60)
        print(f"RETRIEVAL MODE: {search_mode.upper()}")
        print("=" * 60)

        all_results = []

        for item in evaluation_dataset:

            print("\nQuestion:", item["question"])

            documents = retrieval_service.retrieve_documents(
                question=item["question"],
                top_k=k,
                company=item["company"],
                fiscal_year=item["fiscal_year"],
                search_mode=search_mode
            )

            retrieved_ids = [
                document["id"]
                for document in documents
            ]

            relevant_ids = item["relevant_chunk_ids"]

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

        # Calculate average metrics for this mode
        average_metrics = {}

        for metric in [
            "precision_at_k",
            "recall_at_k",
            "f1_at_k"
        ]:

            average = sum(
                result[metric]
                for result in all_results
            ) / len(all_results)

            average_metrics[metric] = average

        overall_results[search_mode] = average_metrics

    # Display final comparison
    print("\n" + "=" * 75)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Search Mode':<15}"
        f"{'Precision@3':<18}"
        f"{'Recall@3':<18}"
        f"{'F1@3':<15}"
    )

    print("-" * 75)

    for mode, metrics in overall_results.items():

        print(
            f"{mode:<15}"
            f"{metrics['precision_at_k']:<18.3f}"
            f"{metrics['recall_at_k']:<18.3f}"
            f"{metrics['f1_at_k']:<15.3f}"
        )


if __name__ == "__main__":
    main()
