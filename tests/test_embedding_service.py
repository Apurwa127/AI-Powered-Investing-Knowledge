
from app.embedding_service import EmbeddingService


def test_embedding_generation():

    embedding_service = EmbeddingService()

    text = "Apple reported strong revenue growth in 2024."

    vector = embedding_service.generate_embedding(text)

    print("Embedding generated successfully!")
    print("Vector dimensions:", len(vector))
    print("First five values:", vector[:5])

    assert len(vector) == 1536


def test_batch_embedding_generation():

    embedding_service = EmbeddingService()

    texts = [
        "Apple reported strong revenue growth in 2024.",
        "Microsoft expanded its cloud computing business.",
        "Tesla increased its vehicle deliveries."
    ]

    vectors = embedding_service.generate_embeddings(texts)

    assert len(vectors) == 3

    for vector in vectors:
        assert len(vector) == 1536


def test_empty_batch():

    embedding_service = EmbeddingService()

    try:
        embedding_service.generate_embeddings([])
    except ValueError as error:
        assert str(error) == "No texts supplied for embedding generation"
    else:
        assert False, "Expected ValueError for empty input"
