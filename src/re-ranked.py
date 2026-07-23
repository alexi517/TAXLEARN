"""
query_reranked.py — RAG query pipeline WITH reranking, using GROQ for generation.

Switched from Gemini to Groq for two reasons:
  1. Gemini free-tier quota was exhausted.
  2. Your BASELINE eval ran on Groq — so using Groq here too means the ONLY
     difference between baseline and reranked is the reranker itself. Clean,
     rigorous before/after comparison.

THE RERANKER IS UNCHANGED — it's a local cross-encoder, no API, no quota.
Only the answer-GENERATION model changed (Gemini -> Groq).

retrieve 20 (bi-encoder) -> rerank (cross-encoder) -> keep 4 -> Groq LLM answers

Run with:  python src/query_reranked.py
"""

import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaGroq          # <-- CHANGED (was Gemini)
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# ---------------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nigeria_tax"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"     # MUST match ingestion
GROQ_MODEL = "llama-3.3-70b-versatile"           # <-- CHANGED (was Gemini model)

RETRIEVE_K = 20      # wide net (bi-encoder, recall)
RERANK_TOP_N = 4     # keep best after reranking (cross-encoder, precision)
RERANKER_MODEL = "BAAI/bge-reranker-base"        # local cross-encoder, no API
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
    # CHANGED: check GROQ key instead of GOOGLE key
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = LlamaGroq(model=GROQ_MODEL)                # <-- CHANGED (was Gemini)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


def build_reranked_query_engine(index):
    # local cross-encoder reranker — unchanged, no API involved
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=RERANK_TOP_N,
    )
    query_engine = index.as_query_engine(
        similarity_top_k=RETRIEVE_K,          # retrieve 20
        node_postprocessors=[reranker],       # rerank down to 4
        text_qa_template=QA_PROMPT,
    )
    return query_engine


def answer_question(query_engine, question: str):
    response = query_engine.query(question)
    sources = []
    for node in response.source_nodes:        # the 4 that survived reranking
        sources.append({
            "file": node.metadata.get("file_name", "unknown"),
            "rerank_score": round(node.score, 3) if node.score else None,
        })
    return str(response), sources


if __name__ == "__main__":
    index = load_index()
    query_engine = build_reranked_query_engine(index)

    print("Nigeria Tax RAG (WITH reranking, Groq) — ask a question ('quit' to exit)\n")
    print(f"Retrieving {RETRIEVE_K} chunks, reranking down to {RERANK_TOP_N}.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        answer, sources = answer_question(query_engine, question)
        print("\n" + "-" * 60)
        print("ANSWER:")
        print(answer)
        print("\nSOURCES (after reranking):")
        for s in sources:
            print(f"  - {s['file']} (rerank score: {s['rerank_score']})")
        print("-" * 60 + "\n")