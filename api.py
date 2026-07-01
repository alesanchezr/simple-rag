"""HTTP API layer. RAG logic lives in rag.py, orchestration in langgraph_layer.py."""

from fastapi import FastAPI
from pydantic import BaseModel

import langgraph_layer
import rag

app = FastAPI()

# Index documents into Qdrant when the server starts.
rag.setup()


class Question(BaseModel):
    query: str


@app.post("/ask")
def ask(body: Question):
    return langgraph_layer.run(body.query)
