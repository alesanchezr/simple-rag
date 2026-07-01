"""
RAG module: Retrieval-Augmented Generation.

This file handles everything related to RAG:
  1. Indexing documents into a vector database (setup)
  2. Finding relevant chunks for a question (retrieve)
  3. Generating an answer with an LLM using that context (query)
"""

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Qdrant stores vectors and lets us search by semantic similarity.
client = QdrantClient(host="localhost", port=6333)
COLLECTION = "documents"
EMBED_MODEL = "nomic-embed-text"  # turns text into a list of numbers (a vector)
CHAT_MODEL = "llama3"             # generates the final answer

# Knowledge base: short texts we want the system to answer questions about.
documents = [
    "4Geeks Academy is a coding bootcamp with campuses in Miami and Spain.",
    "4Geeks courses cover Full Stack, Data Science, and AI Engineering.",
    "LearnPack is 4Geeks' interactive exercises platform.",
    "Rigobot is 4Geeks' AI tutor that guides students.",
]


def embed(text: str) -> list[float]:
    """Convert text to a vector so we can compare meaning, not just keywords."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def setup():
    """
    Index documents into Qdrant on first run.

    Each document is embedded and stored with its original text as metadata.
    768 dimensions matches nomic-embed-text output size.
    """
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=embed(doc), payload={"text": doc})
            for i, doc in enumerate(documents)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        print("Collection created and indexed.")


def retrieve(query: str, limit: int = 2) -> list[str]:
    """
    Retrieval step: find the most relevant document chunks.

    We embed the question and search Qdrant for the closest vectors (cosine similarity).
    """
    vector_query = embed(query)
    results = client.query_points(
        collection_name=COLLECTION,
        query=vector_query,
        limit=limit,
    )
    return [r.payload["text"] for r in results.points]


def query(user_query: str, limit: int = 2) -> dict:
    """
    Full RAG pipeline: retrieve context, then generate an answer.

    The LLM only sees the retrieved chunks — not the entire knowledge base.
    """
    context = "\n".join(retrieve(user_query, limit=limit))

    # Augmentation: inject retrieved context into the prompt.
    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        "Answer:"
    )

    # Generation: LLM produces the final response.
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": resp["message"]["content"],
        "context_used": context,
    }
