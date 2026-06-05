# Homework RAG

A simple MVP RAG project built step by step with:
- dataset preparation
- ingestion
- chunking
- TF-IDF indexing
- retrieval
- grounded answer generation
- Streamlit UI

## Project structure

- data/raw/datasets.json — raw demo dataset
- data/processed/documents.jsonl — ingested documents
- data/processed/chunks.jsonl — chunked documents
- data/index/vectorizer.pkl — saved TF-IDF vectorizer
- data/index/matrix.npz — saved TF-IDF matrix
- data/index/chunks.jsonl — chunks used by retrieval
- app/ — application code
- scripts/ — helper scripts
- tests/ — tests

## Setup

Clone the repository:

git clone https://github.com/EvaNovolaeva/Homework-RAG.git
cd Homework-RAG

Install uv if needed, then create the environment and install dependencies:

uv venv
uv sync

## Prepare data

Check raw dataset file:

uv run python -c "import json; d=json.load(open('data/raw/datasets.json', encoding='utf-8')); print(len(d['datasets']))"

## Ingestion

Generate documents:

uv run python scripts/ingest.py

## Chunking

Run chunking:

uv run python app/chunker.py

Run chunking tests:

PYTHONPATH=. uv run pytest tests/test_chunking.py -v

## Build TF-IDF index

Build the index:

PYTHONPATH=. uv run python scripts/build_index.py

This creates:
- data/index/vectorizer.pkl
- data/index/matrix.npz
- data/index/chunks.jsonl

## Retrieval check

Run retrieval demo:

PYTHONPATH=. uv run python scripts/check_retrieval.py

## Generator check

Run grounded answer demo:

PYTHONPATH=. uv run python scripts/check_generator.py

A relevant query should return an answer with sources.  
An irrelevant query should return a refusal.

## Run tests

Run all tests:

PYTHONPATH=. uv run pytest tests/ -v

## Run Streamlit UI

Start the app:

PYTHONPATH=. uv run python -m streamlit run app/main.py

Then open:

http://localhost:8501

Demo questions:
- How are stress and anxiety related to social media use and sleep?
- Which profiles show high stress and high depression risk?
- What patterns are visible for teenagers with low stress?
- What is the capital of France?

The last one is a negative example and should produce a refusal.

## Notes

This project uses a simple TF-IDF retrieval pipeline for educational purposes.
The generator answers only from retrieved chunks and refuses when relevant context is missing.
