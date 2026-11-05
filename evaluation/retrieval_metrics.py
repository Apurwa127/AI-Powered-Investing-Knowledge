
def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty")

    retrieved_top_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in retrieved_top_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("Relevant IDs cannot be empty")

    retrieved_top_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in retrieved_top_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / len(relevant_ids)

def f1_at_k(precision: float, recall: float) -> float:

    if precision < 0 or recall < 0:
        raise ValueError("Scores cannot be negative")

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> dict:

    precision = precision_at_k(retrieved_ids, relevant_ids, k)
    recall = recall_at_k(retrieved_ids, relevant_ids, k)
    f1 = f1_at_k(precision, recall)

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1
    }

