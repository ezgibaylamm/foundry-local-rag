from pathlib import Path
import json
import fitz

from foundry_local_sdk import (
    Configuration,
    FoundryLocalManager,
)

from src.config import (
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from src.utils import chunk_text

from src.database import (
    initialize_database,
    get_connection,
    count_chunks,
)

from src.embeddings import (
    get_embedding_model,
    generate_embedding,
)


def read_pdf(path: Path) -> str:
    document = fitz.open(path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def save_chunk(
    source_name: str,
    chunk_index: int,
    content: str,
    embedding: list[float],
) -> None:

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO document_chunks
            (
                source_name,
                chunk_index,
                content,
                embedding
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                source_name,
                chunk_index,
                content,
                json.dumps(embedding),
            ),
        )

        connection.commit()


def main() -> None:
    initialize_database()

    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    pdfs = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdfs:
        print("No PDF files found.")
        return

    model = get_embedding_model(manager)
    client = model.get_embedding_client()

    try:

        for pdf in pdfs:
            print(f"\nReading: {pdf.name}")

            text = read_pdf(pdf)

            chunks = chunk_text(
                text,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )

            print(f"Chunks: {len(chunks)}")

            for index, chunk in enumerate(chunks):

                print(
                    f"\rEmbedding chunk "
                    f"{index + 1}/{len(chunks)}",
                    end="",
                    flush=True,
                )

                embedding = generate_embedding(
                    client,
                    chunk,
                )

                save_chunk(
                    source_name=pdf.name,
                    chunk_index=index,
                    content=chunk,
                    embedding=embedding,
                )

            print()

    finally:
        model.unload()
        print("Embedding model unloaded.")

    print(
        f"Stored chunks: {count_chunks()}"
    )


if __name__ == "__main__":
    main()