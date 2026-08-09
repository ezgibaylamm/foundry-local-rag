# Foundry Local RAG

A fully local Retrieval-Augmented Generation (RAG) application built with Python, SQLite, and Microsoft Foundry Local.

The project retrieves relevant information from locally indexed documents using semantic search and provides the retrieved context to a locally running language model to generate grounded answers.

The complete pipeline runs locally without requiring a cloud-based language model API.

---

## Features

- Local PDF document ingestion
- Text extraction and chunking
- SQLite-based document storage
- Local embedding generation
- Semantic similarity search
- Retrieval-Augmented Generation (RAG)
- Local language model inference
- Similarity threshold filtering
- Source and chunk references
- Interactive command-line interface
- Streamlit web interface
- Retrieval evaluation and benchmarking
- CSV evaluation result export
- Embedding model reuse for improved performance

---

## Architecture

The application follows a standard RAG pipeline:

```text
PDF Documents
      |
      v
Text Extraction
      |
      v
Text Chunking
      |
      v
Embedding Generation
      |
      v
SQLite Database
      |
      v
Semantic Retrieval
      |
      v
Relevant Chunks
      |
      v
Local Language Model
      |
      v
Grounded Answer
```

Microsoft Foundry Local is used to run both the embedding model and chat model locally.

---

## Project Structure

```text
FoundryLocalRAG/
│
├── data/
│   ├── rag.db
│   └── evaluation_results.csv
│
├── documents/
│   └── *.pdf
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
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies

- Python
- Microsoft Foundry Local
- SQLite
- Streamlit
- NumPy
- PDF text extraction
- Local embedding models
- Local language models

---

## Models

The current configuration uses:

### Embedding Model

```text
qwen3-embedding-0.6b
```

The embedding model converts document chunks and user queries into vector representations used for semantic retrieval.

### Chat Model

```text
qwen2.5-0.5b
```

The chat model generates answers using the document chunks retrieved by the semantic search pipeline.

Both models run locally through Microsoft Foundry Local.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ezgibaylamm/foundry-local-rag.git
cd foundry-local-rag
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Add documents

Place PDF documents inside:

```text
documents/
```

---

### 2. Ingest documents

Run:

```bash
python -m src.ingest
```

This process:

1. Reads PDF documents
2. Extracts text
3. Splits the text into chunks
4. Generates embeddings locally
5. Stores chunks and embeddings in SQLite

The resulting local database is stored at:

```text
data/rag.db
```

---

### 3. Test semantic retrieval

Run:

```bash
python -m src.retrieval
```

This performs semantic search against the indexed document chunks and displays the most relevant results together with their similarity scores.

---

### 4. Run the command-line RAG assistant

Run:

```bash
python -m src.chat
```

The assistant allows interactive questions about the indexed documents.

Example:

```text
You: What is Retrieval-Augmented Generation?

Assistant: Retrieval-Augmented Generation (RAG) is an AI design pattern...
```

The assistant also displays the source document, chunk number, and similarity score used to generate the answer.

Questions that are not sufficiently supported by the indexed documents are rejected using the configured similarity threshold.

Type:

```text
exit
```

to close the assistant.

---

### 5. Run the Streamlit interface

Start the web application with:

```bash
streamlit run app.py
```

Streamlit will open the application in the browser, typically at:

```text
http://localhost:8501
```

The web interface provides:

- Interactive document-based Q&A
- Semantic document retrieval
- Local language model inference
- Similarity threshold filtering
- Source chunk references
- Conversation history
- Clear conversation control
- Local model status information

---

## Retrieval

User questions are converted into embeddings using the same embedding model used during document ingestion.

Cosine similarity is then calculated between the query embedding and stored document embeddings.

The highest-scoring chunks are selected as context for the language model.

A similarity threshold is used to prevent unrelated questions from being answered using irrelevant document content.

Current threshold:

```text
0.40
```

For example, a question related to the indexed RAG document can be answered, while an unrelated question such as:

```text
What is the capital of Japan?
```

is rejected when the retrieved chunks do not meet the required similarity threshold.

---

## Evaluation

The project includes a retrieval evaluation script.

Run:

```bash
python -m src.evaluate
```

The evaluation contains both:

- Answerable document-related questions
- Unanswerable out-of-domain questions

For each test, the system records:

- Question
- Best similarity score
- Expected answerability
- Predicted answerability
- Retrieval time
- Pass/fail result

The evaluation summary reports:

```text
Passed tests
Accuracy
Total retrieval time
Average retrieval time
```

Evaluation results are also exported to:

```text
data/evaluation_results.csv
```

---

## Performance Optimization

The embedding model is reused during evaluation rather than being repeatedly initialized for every query.

This significantly reduces retrieval latency and makes repeated semantic searches more efficient.

The evaluation pipeline can therefore be used for both correctness testing and basic retrieval performance benchmarking.

---

## Local-First Design

The project is designed around local execution.

Document processing, embeddings, retrieval, and answer generation are performed locally.

This provides several advantages:

- No cloud LLM API is required
- Documents remain on the local machine
- Reduced dependency on external services
- Local experimentation with embedding and language models
- Greater control over the complete RAG pipeline

---

## Current Status

The following components are implemented:

- PDF ingestion
- Text chunking
- SQLite document storage
- Local embedding generation
- Semantic retrieval
- Cosine similarity ranking
- Similarity threshold filtering
- Local chat model integration
- Retrieval-Augmented Generation
- Interactive CLI assistant
- Source references
- Retrieval evaluation
- Performance measurement
- CSV evaluation export
- Embedding model reuse
- Streamlit web interface

The core Local RAG pipeline is functional end-to-end.

---

## Possible Future Improvements

Possible extensions include:

- PDF upload directly from the web interface
- Multiple document collections
- Configurable Top-K retrieval
- Configurable similarity threshold
- Additional retrieval evaluation metrics
- Persistent conversation history
- More extensive performance benchmarking
- Support for additional local embedding and chat models
- Improved document management

---

## Example Workflow

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ingest documents
python -m src.ingest

# Test retrieval
python -m src.retrieval

# Run evaluation
python -m src.evaluate

# Run CLI assistant
python -m src.chat

# Run web interface
streamlit run app.py
```

---

## Summary

This project demonstrates an end-to-end local Retrieval-Augmented Generation pipeline using Microsoft Foundry Local.

It combines local document ingestion, embeddings, SQLite storage, semantic retrieval, similarity filtering, local language model inference, evaluation, and a Streamlit user interface into a single document-grounded question-answering application.