# Foundry Local RAG

A fully local Retrieval-Augmented Generation (RAG) application built with Python, Microsoft Foundry Local, SQLite, and local language models.

The project demonstrates how documents can be processed, embedded, stored, retrieved, and used as context for a local LLM without relying on a cloud-based inference service.

## Features

- Fully local model inference with Foundry Local
- PDF document ingestion
- Text extraction and chunking
- Local embedding generation
- SQLite-based document and embedding storage
- Semantic search using cosine similarity
- Top-K document retrieval
- Similarity threshold for out-of-scope questions
- Local LLM response generation
- Source and chunk references
- Interactive command-line RAG assistant
- Retrieval evaluation
- CSV evaluation results
- Persistent model loading for improved performance

## Project Structure

```text
FoundryLocalRAG/
│
├── data/
│   ├── rag.db
│   └── evaluation_results.csv
│
├── documents/
│   └── Summer School Foundry Local Plan.pdf
│
├── models/
│
├── src/
│   ├── __init__.py
│   ├── chat.py
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── evaluate.py
│   ├── ingest.py
│   ├── retrieval.py
│   └── utils.py
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## RAG Pipeline

The application follows the standard Retrieval-Augmented Generation workflow:

```text
PDF Documents
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embedding Generation
      ↓
SQLite Storage
      ↓
User Question
      ↓
Query Embedding
      ↓
Cosine Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
Similarity Threshold
      ↓
Local LLM
      ↓
Grounded Answer + Sources
```

## Models

The project currently uses:

### Embedding Model

```text
qwen3-embedding-0.6b
```

The embedding model converts document chunks and user queries into vector representations.

The generated embeddings contain 1024 dimensions.

### Chat Model

```text
qwen2.5-0.5b
```

The chat model receives the retrieved document context and generates the final answer locally.

## Document Ingestion

Documents placed inside the `documents/` directory can be processed by the ingestion pipeline.

Run:

```bash
python -m src.ingest
```

The ingestion process:

1. Reads the PDF document
2. Extracts text
3. Splits the text into chunks
4. Generates an embedding for each chunk
5. Stores the chunk and embedding in SQLite

Example:

```text
Reading: Summer School Foundry Local Plan.pdf
Chunks: 71
Embedding chunk 71/71
Stored chunks: 71
```

## Semantic Retrieval

The retrieval system converts a user query into an embedding and compares it with stored document embeddings using cosine similarity.

The most relevant chunks are returned using Top-K retrieval.

Run:

```bash
python -m src.retrieval
```

Example query:

```text
What is Retrieval-Augmented Generation?
```

Example retrieval:

```text
Result 1
Chunk: 8
Similarity: 0.6798

Result 2
Chunk: 18
Similarity: 0.6353

Result 3
Chunk: 27
Similarity: 0.6283
```

## Interactive RAG Assistant

Start the local assistant with:

```bash
python -m src.chat
```

Example:

```text
==============================
     Local RAG Assistant
==============================

Ask questions about your documents.
Type 'exit' to quit.

You: What are embeddings used for?

Assistant: ...

Sources:
1. Summer School Foundry Local Plan.pdf (Chunk 26, Similarity 0.6385)
2. Summer School Foundry Local Plan.pdf (Chunk 27, Similarity 0.6354)
3. Summer School Foundry Local Plan.pdf (Chunk 29, Similarity 0.6216)

You:
```

The assistant generates answers using retrieved document context and displays the source chunks used during retrieval.

## Out-of-Scope Questions

A similarity threshold is used to prevent the model from answering questions that are not supported by the document collection.

Current threshold:

```text
0.40
```

For example:

```text
You: What is the capital of Japan?

Assistant: I don't know based on the provided documents.
```

This reduces unsupported answers and helps keep responses grounded in the local knowledge base.

## Evaluation

The retrieval system includes a small evaluation suite containing both answerable and out-of-scope questions.

Run:

```bash
python -m src.evaluate
```

Current evaluation result:

```text
Passed: 7/7
Accuracy: 100.0%
```

The evaluation checks whether the similarity threshold correctly separates questions supported by the document from unrelated questions.

Results are also stored in:

```text
data/evaluation_results.csv
```

## Performance Optimization

Initially, the embedding model was loaded and unloaded for every retrieval query.

This resulted in approximately:

```text
Total retrieval time: 12.417 seconds
Average retrieval time: 1.774 seconds
```

The pipeline was optimized by loading the embedding model once and reusing the same embedding client across multiple queries.

After optimization:

```text
Total retrieval time: 0.486 seconds
Average retrieval time: 0.069 seconds
```

This reduced the measured average retrieval time by approximately 96%.

The interactive assistant also keeps both the embedding model and chat model loaded for the entire chat session.

The models are unloaded only when the user exits the application.

## Installation

Clone the repository:

```bash
git clone https://github.com/ezgibaylamm/foundry-local-rag.git
cd foundry-local-rag
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Ingest documents

```bash
python -m src.ingest
```

### 2. Test retrieval

```bash
python -m src.retrieval
```

### 3. Run evaluation

```bash
python -m src.evaluate
```

### 4. Start the RAG assistant

```bash
python -m src.chat
```

## Technology Stack

- Python
- Microsoft Foundry Local
- Qwen embedding model
- Qwen local chat model
- SQLite
- Cosine similarity
- Retrieval-Augmented Generation (RAG)

## Current Status

The core local RAG pipeline is functional.

Implemented components include:

- Document ingestion
- Chunk generation
- Embedding generation
- Local vector storage
- Semantic retrieval
- Similarity filtering
- Local LLM generation
- Source attribution
- Interactive CLI
- Retrieval evaluation
- Performance optimization

## Next Steps

Planned improvements include:

- Streamlit web interface
- Support for multiple uploaded documents
- Improved document management
- Retrieval configuration controls
- Better evaluation metrics
- Chat history
- Additional performance benchmarking