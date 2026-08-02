from src.database import DATABASE_PATH, count_chunks, initialize_database


def main() -> None:
    initialize_database()

    print("Foundry Local RAG")
    print(f"Database: {DATABASE_PATH}")
    print(f"Stored chunks: {count_chunks()}")


if __name__ == "__main__":
    main()