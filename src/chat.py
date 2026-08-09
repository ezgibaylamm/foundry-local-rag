from foundry_local_sdk import Configuration, FoundryLocalManager

from src.retrieval import get_top_chunks


CHAT_MODEL_ALIAS = "qwen2.5-0.5b"


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
) -> str:
    """
    Kullanıcı sorusu için en alakalı chunk'ları bulur
    ve local chat modeli ile cevap üretir.
    """

    results = get_top_chunks(
        question,
        manager,
        top_k=3,
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
                "If the context does not contain enough "
                "information, say that you don't know "
                "based on the provided documents.\n\n"
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

        print("\n")

        return "".join(answer_parts)

    finally:
        model.unload()


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
            answer_query(
                question,
                manager,
            )

        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()