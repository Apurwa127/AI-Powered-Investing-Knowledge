
from app.embedding_service import EmbeddingService
from app.retrieval_service import RetrievalService
from app.reranker import Reranker

from evaluation.evaluation_dataset import evaluation_dataset
from evaluation.retrieval_metrics import evaluate_retrieval


def main():

    # Initialize services
    embedding_service = EmbeddingService()
    retrieval_service = RetrievalService(embedding_service)
    reranker = Reranker()

    # Evaluation configuration
    k = 3
    retrieval_k = 10

    baseline_results = []
    reranked_results = []

    # Evaluate every question
    for item in evaluation_dataset:

        print("\n" + "=" * 75)
        print("Question:", item["question"])
        print("Company:", item["company"])

        # ------------------------------------------------
        # Experiment A: Hybrid retrieval without reranking
        # ------------------------------------------------

        baseline_documents = retrieval_service.retrieve_documents(
            question=item["question"],
            top_k=k,
            company=item["company"],
            fiscal_year=item["fiscal_year"],
            search_mode="hybrid"
        )

        # ------------------------------------------------
        # Experiment B: Hybrid retrieval with reranking
        # ------------------------------------------------

        candidate_documents = retrieval_service.retrieve_documents(
            question=item["question"],
            top_k=retrieval_k,
            company=item["company"],
            fiscal_year=item["fiscal_year"],
            search_mode="hybrid"
        )

        # Rerank all retrieved candidates
        all_ranked_documents = reranker.rerank(
            question=item["question"],
            documents=candidate_documents,
            top_k=len(candidate_documents)
        )

        # ------------------------------------------------
        # Display reranking diagnostics
        # ------------------------------------------------

        print("\nRERANKING DIAGNOSTICS")

        for rank, document in enumerate(
            all_ranked_documents,
            start=1
        ):

            print(f"\nRank: {rank}")
            print("ID:", document["id"])
            print("Page:", document["page_number"])
            print("Score:", document["rerank_score"])
            print("Content:", document["content"][:500])

        # Select top 3 reranked documents
        reranked_documents = all_ranked_documents[:k]

        # ------------------------------------------------
        # Prepare ground-truth IDs
        # ------------------------------------------------

        relevant_ids = item["relevant_chunk_ids"]

        baseline_ids = [
            document["id"]
            for document in baseline_documents
        ]

        reranked_ids = [
            document["id"]
            for document in reranked_documents
        ]

        # ------------------------------------------------
        # Calculate baseline metrics
        # ------------------------------------------------

        baseline_metrics = evaluate_retrieval(
            retrieved_ids=baseline_ids,
            relevant_ids=relevant_ids,
            k=k
        )

        # ------------------------------------------------
        # Calculate reranked metrics
        # ------------------------------------------------

        reranked_metrics = evaluate_retrieval(
            retrieved_ids=reranked_ids,
            relevant_ids=relevant_ids,
            k=k
        )

        baseline_results.append(baseline_metrics)
        reranked_results.append(reranked_metrics)

        # ------------------------------------------------
        # Display individual question results
        # ------------------------------------------------

        print("\nBaseline retrieved IDs:")
        print(baseline_ids)

        print("\nReranked IDs:")
        print(reranked_ids)

        print("\nBaseline metrics:")
        print(baseline_metrics)

        print("\nReranked metrics:")
        print(reranked_metrics)

    # ------------------------------------------------
    # Calculate average evaluation metrics
    # ------------------------------------------------

    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "f1_at_k"
    ]

    print("\n" + "=" * 75)
    print("RETRIEVAL VS RERANKING COMPARISON")
    print("=" * 75)

    print(
        f"{'Metric':<20}"
        f"{'Hybrid':<18}"
        f"{'Hybrid + Reranking':<22}"
    )

    print("-" * 60)

    for metric in metric_names:

        baseline_average = sum(
            result[metric]
            for result in baseline_results
        ) / len(baseline_results)

        reranked_average = sum(
            result[metric]
            for result in reranked_results
        ) / len(reranked_results)

        print(
            f"{metric:<20}"
            f"{baseline_average:<18.3f}"
            f"{reranked_average:<22.3f}"
        )


if __name__ == "__main__":
    main()
