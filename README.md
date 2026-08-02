# Foundry Local RAG

An offline Retrieval-Augmented Generation (RAG) assistant built with Microsoft Foundry Local, Python, SQLite, and local language models.

## Project Goal

The goal of this project is to build a fully local document question-answering assistant. The application will retrieve relevant information from local documents and use an on-device language model to generate grounded answers.

## Current Status

- [x] Python virtual environment created
- [x] Foundry Local SDK installed
- [x] Foundry Local CLI tested
- [x] Local Qwen3 1.7B model downloaded
- [x] Local GPU inference verified
- [ ] Document ingestion
- [ ] Text chunking
- [ ] Embedding generation
- [ ] SQLite storage
- [ ] Semantic retrieval
- [ ] RAG question-answering pipeline
- [ ] Testing and documentation

## Planned Architecture

1. Load local documents
2. Split documents into chunks
3. Generate embeddings
4. Store chunks and embeddings in SQLite
5. Retrieve relevant chunks for a user question
6. Send the retrieved context to a Foundry Local model
7. Generate a source-grounded answer

## Technology Stack

- Python
- Microsoft Foundry Local
- SQLite
- Qwen3 1.7B