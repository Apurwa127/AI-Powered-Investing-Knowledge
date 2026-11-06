from app.reranker import Reranker


def main():

    reranker = Reranker()

    question = "What was Apple's net income in 2024?"

    documents = [
        {
            "id": "chunk_1",
            "content": "Apple reported total revenue of $391.035 billion in 2024."
        },
        {
            "id": "chunk_2",
            "content": "Apple reported net income of $93.736 billion in 2024."
        },
        {
            "id": "chunk_3",
            "content": "Apple's research and development expenses were $31.370 billion."
        }
    ]

    results = reranker.rerank(
        question=question,
        documents=documents,
        top_k=2
    )

    for document in results:
        print("ID:", document["id"])
        print("Rerank score:", document["rerank_score"])
        print("Content:", document["content"])
        print("-" * 60)


if __name__ == "__main__":
    main()