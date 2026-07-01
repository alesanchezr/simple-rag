"""
LangGraph orchestration layer.

Defines the RAG workflow as a graph: retrieve → generate.
RAG primitives (embeddings, vector search, LLM calls) live in rag.py.

Note: this file is named langgraph_layer.py (not langgraph.py) to avoid
shadowing the installed `langgraph` package on import.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import rag


class RAGState(TypedDict):
    query: str
    limit: int
    context: str
    answer: str


def retrieve_node(state: RAGState) -> dict:
    """Node 1: fetch relevant chunks from the vector database."""
    chunks = rag.retrieve(state["query"], limit=state["limit"])
    return {"context": "\n".join(chunks)}


def generate_node(state: RAGState) -> dict:
    """Node 2: augment the prompt with context and call the LLM."""
    answer = rag.generate(state["query"], state["context"])
    return {"answer": answer}


# Build the graph: START → retrieve → generate → END
_builder = StateGraph(RAGState)
_builder.add_node("retrieve", retrieve_node)
_builder.add_node("generate", generate_node)
_builder.add_edge(START, "retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", END)

graph = _builder.compile()


def run(user_query: str, limit: int = 2) -> dict:
    """Run the RAG graph and return the same shape as rag.query()."""
    result = graph.invoke(
        {
            "query": user_query,
            "limit": limit,
            "context": "",
            "answer": "",
        }
    )
    return {
        "answer": result["answer"],
        "context_used": result["context"],
    }
