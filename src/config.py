from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "documents"

DATA_DIR = BASE_DIR / "data"

DATABASE_PATH = DATA_DIR / "rag.db"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 100