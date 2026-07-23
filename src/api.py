"""
api.py — FastAPI wrapper around the hybrid RAG pipeline.

This turns your RAG system from "a script I run in a terminal" into "a web
service other programs and users can send questions to."

THE ONE ENDPOINT: POST /query
  - receives:  {"question": "What is the Tertiary Education Tax rate?"}
  - runs:      your existing hybrid + rerank RAG pipeline
  - returns:   {"answer": "...", "sources": [...]}

REQUEST FLOW:
  1. A request arrives at /query carrying a question (as JSON).
  2. FastAPI validates it and hands it to the query() function below.
  3. That function runs your RAG pipeline (same logic as query_hybrid.py).
  4. It returns the answer; FastAPI packages it back as JSON.

The RAG ENGINE is unchanged — we import and reuse it. This file is only the
"doorway."

RUN THE SERVER:
  uvicorn src.api:app --reload
Then open http://127.0.0.1:8000/docs  (FastAPI's auto-generated test page)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

# Reuse the RAG pipeline you already built — don't rewrite it.
from src.query_hybrid import (
    load_index_and_collection,
    build_hybrid_query_engine,
    answer_question,
)

# ---------------------------------------------------------------------------
# Request/response "shapes" — Pydantic models.
# These tell FastAPI what a valid request looks like and what we send back.
# FastAPI uses them to validate input and auto-document the API.
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """What a caller must send: a question string."""
    question: str


class Source(BaseModel):
    file: str
    score: float | None = None


class QueryResponse(BaseModel):
    """What we send back: the answer plus the sources used."""
    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Load the RAG engine ONCE at startup, not per request.
# Building the index + reranker is slow; doing it on every request would make
# the API crawl. We build it once when the server starts and reuse it.
# ---------------------------------------------------------------------------

# a simple holder for the loaded query engine
state = {"query_engine": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once when the server starts up
    print("Loading RAG engine (index, BM25, reranker) ...")
    index, chroma_collection = load_index_and_collection()
    state["query_engine"] = build_hybrid_query_engine(index, chroma_collection)
    print("RAG engine ready.")
    yield
    # (nothing to clean up on shutdown for now)


app = FastAPI(
    title="Nigeria Tax RAG API",
    description="Ask questions about Nigerian 2025 tax law. Answers are grounded "
                "in official documents with citations.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    """A simple 'is the server alive?' endpoint. Visiting the root URL hits this."""
    return {"status": "ok", "message": "Nigeria Tax RAG API is running."}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """The main endpoint: takes a question, returns a grounded answer + sources."""
    engine = state["query_engine"]
    if engine is None:
        return {"answer": "Engine not loaded yet.", "sources": []}

    # THIS is your existing RAG logic — unchanged, just called from the web layer.
    answer, sources = answer_question(engine, request.question)

    return QueryResponse(
        answer=answer,
        sources=[Source(file=s["file"], score=s.get("score")) for s in sources],
    )


# ---------------------------------------------------------------------------
# requirements.txt additions:
#   fastapi
#   uvicorn[standard]
#   pydantic   (usually already installed as a FastAPI dependency)
# ---------------------------------------------------------------------------