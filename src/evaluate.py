from foundry_local_sdk import Configuration, FoundryLocalManager

from src.retrieval import get_top_chunks


SIMILARITY_THRESHOLD = 0.40


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


def main() -> None:
    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    passed = 0

    print("\n==============================")
    print("      RAG Evaluation")
    print("==============================\n")

    for index, test_case in enumerate(TEST_CASES, start=1):
        question = test_case["question"]
        expected = test_case["should_be_answerable"]

        results = get_top_chunks(
            question,
            manager,
            top_k=3,
        )

        best_score = (
            results[0]["score"]
            if results
            else 0.0
        )

        predicted = best_score >= SIMILARITY_THRESHOLD

        success = predicted == expected

        if success:
            passed += 1

        print(f"Test {index}")
        print(f"Question: {question}")
        print(f"Best similarity: {best_score:.4f}")
        print(f"Expected answerable: {expected}")
        print(f"Predicted answerable: {predicted}")
        print(f"Result: {'PASS' if success else 'FAIL'}")
        print("-" * 60)

    total = len(TEST_CASES)

    print("\nEvaluation Summary")
    print(f"Passed: {passed}/{total}")
    print(f"Accuracy: {(passed / total) * 100:.1f}%")


if __name__ == "__main__":
    main()