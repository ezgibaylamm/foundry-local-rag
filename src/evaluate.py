import csv
import time
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.embeddings import get_embedding_model
from src.retrieval import get_top_chunks_with_client


SIMILARITY_THRESHOLD = 0.40

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_PATH = DATA_DIR / "evaluation_results.csv"


TEST_CASES = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "should_be_answerable": True,
    },
    {
        "question": "What is Foundry Local?",
        "should_be_answerable": True,
    },
    {
        "question": "Why is SQLite used in this project?",
        "should_be_answerable": True,
    },
    {
        "question": "What are embeddings used for?",
        "should_be_answerable": True,
    },
    {
        "question": "What is the capital of Japan?",
        "should_be_answerable": False,
    },
    {
        "question": "Who won the FIFA World Cup in 2022?",
        "should_be_answerable": False,
    },
    {
        "question": "What is the population of Italy?",
        "should_be_answerable": False,
    },
]


def save_results(rows: list[dict]) -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "question",
        "best_similarity",
        "expected_answerable",
        "predicted_answerable",
        "retrieval_time_seconds",
        "result",
    ]

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    model = get_embedding_model(manager)
    embedding_client = model.get_embedding_client()

    passed = 0
    total_time = 0.0
    evaluation_rows = []

    print("\n==============================")
    print("      RAG Evaluation")
    print("==============================\n")

    try:
        for index, test_case in enumerate(
            TEST_CASES,
            start=1,
        ):
            question = test_case["question"]
            expected = test_case["should_be_answerable"]

            start_time = time.perf_counter()

            results = get_top_chunks_with_client(
                question,
                embedding_client,
                top_k=3,
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            total_time += elapsed_time

            best_score = (
                results[0]["score"]
                if results
                else 0.0
            )

            predicted = (
                best_score
                >= SIMILARITY_THRESHOLD
            )

            success = predicted == expected

            if success:
                passed += 1

            result_label = (
                "PASS"
                if success
                else "FAIL"
            )

            evaluation_rows.append(
                {
                    "question": question,
                    "best_similarity": f"{best_score:.4f}",
                    "expected_answerable": expected,
                    "predicted_answerable": predicted,
                    "retrieval_time_seconds": f"{elapsed_time:.3f}",
                    "result": result_label,
                }
            )

            print(f"Test {index}")
            print(f"Question: {question}")
            print(
                f"Best similarity: "
                f"{best_score:.4f}"
            )
            print(
                f"Expected answerable: "
                f"{expected}"
            )
            print(
                f"Predicted answerable: "
                f"{predicted}"
            )
            print(
                f"Retrieval time: "
                f"{elapsed_time:.3f} seconds"
            )
            print(f"Result: {result_label}")
            print("-" * 60)

    finally:
        model.unload()
        print("\nEmbedding model unloaded.")

    total = len(TEST_CASES)

    accuracy = (
        passed / total
    ) * 100

    average_time = (
        total_time / total
    )

    save_results(
        evaluation_rows
    )

    print("\n==============================")
    print("      Evaluation Summary")
    print("==============================")
    print(f"Passed: {passed}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(
        f"Total retrieval time: "
        f"{total_time:.3f} seconds"
    )
    print(
        f"Average retrieval time: "
        f"{average_time:.3f} seconds"
    )
    print(
        f"Results saved to: "
        f"{RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()