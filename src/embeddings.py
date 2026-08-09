from foundry_local_sdk import Configuration, FoundryLocalManager


EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"


def get_embedding_model(manager: FoundryLocalManager):
    """
    Embedding modelini indirir ve yükler.
    """
    print(f"Preparing embedding model: {EMBEDDING_MODEL_ALIAS}")

    model = manager.catalog.get_model(EMBEDDING_MODEL_ALIAS)

    model.download(
        lambda progress: print(
            f"\rDownloading embedding model: {progress:.1f}%",
            end="",
            flush=True,
        )
    )
    print()

    model.load()
    print("Embedding model loaded successfully.")

    return model


def generate_embedding(client, text: str) -> list[float]:
    """
    Verilen metin için embedding üretir.
    """
    response = client.generate_embedding(text)

    return response.data[0].embedding


def main() -> None:
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    model = get_embedding_model(manager)
    client = model.get_embedding_client()

    try:
        embedding = generate_embedding(
            client,
            "Retrieval-Augmented Generation uses retrieved documents as context.",
        )

        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

    finally:
        model.unload()
        print("Embedding model unloaded.")


if __name__ == "__main__":
    main()