# Simple RAG

Minimal RAG API with FastAPI, Ollama, Qdrant, and LangGraph.

## Project structure

- `rag.py` — RAG primitives (embed, index, retrieve, generate)
- `langgraph_layer.py` — LangGraph orchestration (retrieve → generate)
- `api.py` — HTTP layer

## Requirements

- Python 3.11+
- [Docker](https://www.docker.com/) (for Qdrant)
- [Ollama](https://ollama.com/) running locally

## Setup

### 1. Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

### 2. Qdrant

```bash
docker compose up -d
```

### 3. Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn api:app --reload
```

API runs at http://localhost:8000 (docs at http://localhost:8000/docs).

## Try it

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Rigobot?"}'
```
