from pathlib import Path
import fitz

from src.config import DOCUMENTS_DIR
from src.utils import chunk_text
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def read_pdf(path: Path) -> str:
    document = fitz.open(path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


if __name__ == "__main__":

    pdfs = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdfs:
        print("No PDF files found.")
        exit()

    pdf = pdfs[0]

    print(f"Reading: {pdf.name}")

    text = read_pdf(pdf)

    print(f"\nCharacters: {len(text)}")

    chunks = chunk_text(
        text,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    print(f"Chunks: {len(chunks)}")

    print("\nFirst Chunk:\n")
    print(chunks[0][:1000])