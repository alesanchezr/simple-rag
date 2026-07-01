"""HTTP API layer. RAG logic lives in rag.py."""

from fastapi import FastAPI
from pydantic import BaseModel

import rag

app = FastAPI()

# Index documents into Qdrant when the server starts.
rag.setup()


class Question(BaseModel):
    query: str


@app.post("/ask")
def ask(body: Question):
    return rag.query(body.query)
