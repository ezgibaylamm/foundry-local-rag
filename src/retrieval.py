import json
import math

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.database import get_connection
from src.embeddings import get_embedding_model, generate_embedding


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def get_top_chunks(
    query: str,
    manager: FoundryLocalManager,
    top_k: int = 3,
):
    model = get_embedding_model(manager)
    client = model.get_embedding_client()

    try:
        query_embedding = generate_embedding(
            client,
            query,
        )

        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    source_name,
                    chunk_index,
                    content,
                    embedding
                FROM document_chunks
                """
            ).fetchall()

        results = []

        for row in rows:
            chunk_embedding = json.loads(
                row["embedding"]
            )

            score = cosine_similarity(
                query_embedding,
                chunk_embedding,
            )

            results.append(
                {
                    "source_name": row["source_name"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "score": score,
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]

    finally:
        model.unload()
        print("Embedding model unloaded.")


def main() -> None:
    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    query = "What is the capital of Japan?"

    results = get_top_chunks(
        query,
        manager,
        top_k=3,
    )

    print(f"\nQuery: {query}\n")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(f"Result {index}")
        print(
            f"Source: {result['source_name']}"
        )
        print(
            f"Chunk: {result['chunk_index']}"
        )
        print(
            f"Similarity: {result['score']:.4f}"
        )
        print(result["content"][:500])
        print("-" * 60)


if __name__ == "__main__":
    main()