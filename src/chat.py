from foundry_local_sdk import Configuration, FoundryLocalManager

from src.retrieval import get_top_chunks


CHAT_MODEL_ALIAS = "qwen2.5-0.5b"
SIMILARITY_THRESHOLD = 0.40


def get_chat_model(manager: FoundryLocalManager):
    """
    Chat modelini indirir ve yükler.
    """
    print(f"Preparing chat model: {CHAT_MODEL_ALIAS}")

    model = manager.catalog.get_model(CHAT_MODEL_ALIAS)

    model.download(
        lambda progress: print(
            f"\rDownloading chat model: {progress:.1f}%",
            end="",
            flush=True,
        )
    )
    print()

    model.load()
    print("Chat model loaded successfully.")

    return model


def answer_query(
    question: str,
    manager: FoundryLocalManager,
) -> tuple[str, list[dict]]:
    """
    Kullanıcı sorusu için en alakalı chunk'ları bulur.
    Yeterince alakalı context varsa local chat modeli ile cevap üretir.
    Cevapla birlikte kullanılan kaynakları da döndürür.
    """

    results = get_top_chunks(
        question,
        manager,
        top_k=3,
    )

    if not results:
        return (
            "I don't know based on the provided documents.",
            [],
        )

    best_score = results[0]["score"]

    if best_score < SIMILARITY_THRESHOLD:
        return (
            "I don't know based on the provided documents.",
            [],
        )

    context = "\n\n".join(
        f"[Source: {result['source_name']}, "
        f"Chunk: {result['chunk_index']}]\n"
        f"{result['content']}"
        for result in results
    )

    model = get_chat_model(manager)
    client = model.get_chat_client()

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only "
                "the provided context. "
                "Do not use outside knowledge. "
                "If the context does not contain enough "
                "information to answer the question, say exactly: "
                "\"I don't know based on the provided documents.\" "
                "Keep the answer clear and concise.\n\n"
                f"Context:\n{context}"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    answer_parts = []

    try:
        print("\nAssistant: ", end="", flush=True)

        for chunk in client.complete_streaming_chat(messages):
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if not delta:
                continue

            content = delta.content

            if content:
                answer_parts.append(content)

                print(
                    content,
                    end="",
                    flush=True,
                )

        print()

        answer = "".join(answer_parts)

        return answer, results

    finally:
        model.unload()


def print_sources(results: list[dict]) -> None:
    """
    Cevap oluşturulurken kullanılan kaynakları terminalde gösterir.
    """

    if not results:
        return

    print("\nSources:")

    for index, result in enumerate(results, start=1):
        print(
            f"{index}. {result['source_name']} "
            f"(Chunk {result['chunk_index']}, "
            f"Similarity {result['score']:.4f})"
        )

    print()


def main() -> None:
    """
    Terminal üzerinden çalışan Local RAG chatbot.
    """

    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    print("\n==============================")
    print("     Local RAG Assistant")
    print("==============================")
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            answer, sources = answer_query(
                question,
                manager,
            )

            if not sources:
                print(f"\nAssistant: {answer}\n")
                continue

            print_sources(sources)

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()