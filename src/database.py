import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "rag.db"


def get_connection() -> sqlite3.Connection:
    """
    SQLite veritabanına bağlantı oluşturur.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    """
    RAG uygulaması için gerekli tabloları oluşturur.
    """
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_source_chunk
            ON document_chunks (source_name, chunk_index)
            """
        )

        connection.commit()


def count_chunks() -> int:
    """
    Veritabanındaki toplam chunk sayısını döndürür.
    """
    with get_connection() as connection:
        result = connection.execute(
            "SELECT COUNT(*) AS total FROM document_chunks"
        ).fetchone()

    return int(result["total"])


if __name__ == "__main__":
    initialize_database()

    print(f"Database created successfully: {DATABASE_PATH}")
    print(f"Stored chunks: {count_chunks()}")