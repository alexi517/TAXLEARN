"""
query.py — Retrieval + Generation for the Nigeria Tax RAG project.

This is the step that turns a RETRIEVER (finds chunks) into a RAG SYSTEM
(answers questions using those chunks, with citations).

WHAT HAPPENS PER QUESTION:
  1. Load the existing Chroma vector store (built by ingest.py — no re-embedding)
  2. Retrieve the top-k most relevant chunks for the question
  3. Build a prompt: instruct the LLM to answer ONLY from those chunks + cite sources
  4. Call Gemini (free tier) to generate the answer
  5. Return the answer plus the source documents it used

KEY QUALITY BEHAVIOR: if the answer isn't in the retrieved chunks, the system
must say "I don't have that information" rather than hallucinate. You'll measure
this later in your eval harness (faithfulness).

SETUP:
  - GOOGLE_API_KEY in your .env (free key from https://aistudio.google.com)
  - pip install llama-index-llms-google-genai

Run with:  python src/query.py
"""

import os
import sys

# Windows consoles default to a legacy codepage (e.g. cp1252) that can't
# encode characters like '≤' that the LLM may produce; force UTF-8 output.
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# Load environment variables from .env (your API key)
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nigeria_tax"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # MUST match what ingest.py used
GEMINI_MODEL = "models/gemini-flash-lite-latest"  # free-tier friendly; gemini-flash-latest hit a 20/day cap
TOP_K = 4                                       # how many chunks to retrieve

# ---------------------------------------------------------------------------
# PROMPT — this is where you control answer quality and anti-hallucination.
# Notice the explicit instructions: answer ONLY from context, cite sources,
# and say so if the answer isn't present. Prompt design is a real skill —
# be ready to explain why each instruction is here.
# ---------------------------------------------------------------------------
QA_PROMPT = PromptTemplate(
    "You are a helpful assistant answering questions about Nigerian tax law.\n"
    "Use ONLY the context information below to answer the question.\n"
    "If the answer is not contained in the context, say exactly: "
    "\"I don't have enough information in the provided documents to answer that.\"\n"
    "Do not use outside knowledge. Always cite the source document name for any fact.\n"
    "\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer (with source citations):"
)


def load_index():
    """Reconnect to the vector store that ingest.py already built."""
    if not os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com "
            "and add it to your .env file."
        )

    # The embedding model here MUST be the same one used during ingestion,
    # otherwise the query vector won't match the stored vectors.
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    # The LLM used for generating answers.
    Settings.llm = GoogleGenAI(model=GEMINI_MODEL)

    # Reconnect to the existing Chroma collection (do NOT rebuild it).
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Build an index object from the existing store (fast — no re-embedding).
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


def answer_question(index, question: str):
    """Retrieve context, generate an answer, and return answer + sources."""
    # A query engine ties retrieval + prompt + LLM together.
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=QA_PROMPT,
    )

    response = query_engine.query(question)

    # Collect which source documents were used, for transparency/citations.
    sources = []
    for node in response.source_nodes:
        sources.append({
            "file": node.metadata.get("file_name", "unknown"),
            "score": round(node.score, 3) if node.score else None,
        })

    return str(response), sources


if __name__ == "__main__":
    index = load_index()

    print("Nigeria Tax RAG — ask a question (type 'quit' to exit)\n")

    # A few starter questions to test with:
    #   - "Who is exempt from personal income tax?"
    #   - "What is the rate of Tertiary Education Tax?"
    #   - "What is the capital of France?"  <- should refuse (not in docs!)

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        answer, sources = answer_question(index, question)
        print("\n" + "-" * 60)
        print("ANSWER:")
        print(answer)
        print("\nSOURCES USED:")
        for s in sources:
            print(f"  - {s['file']} (relevance: {s['score']})")
        print("-" * 60 + "\n")
