"""
query_hybrid.py — HYBRID retrieval + reranking (CORRECTED).

Fix vs previous version: BM25 needs raw text nodes, but they live in CHROMA,
not in the in-memory docstore (which was empty — that caused the
"pass exactly one of index, nodes, or docstore" error). So we now fetch the
stored chunks back out of Chroma and hand those to BM25.

    Question
       ├──► Vector retriever (semantic) ──┐
       │                                   ├─► Fusion (RRF) ─► Reranker ─► top 4 ─► LLM
       └──► BM25 retriever (keyword) ─────┘

Run with:  python src/query_hybrid.py
"""

import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# ---------------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nigeria_tax"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "llama-3.3-70b-versatile"

RETRIEVER_K = 10
RERANK_TOP_N = 4
RERANKER_MODEL = "BAAI/bge-reranker-base"
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


def load_index_and_collection():
    """Return both the vector index AND the raw chroma collection.
    We need the collection to pull text nodes back out for BM25."""
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not set in .env")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = LlamaGroq(model=GROQ_MODEL)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index, chroma_collection


def get_nodes_from_chroma(chroma_collection):
    """Fetch all stored chunks (text + metadata) back out of Chroma as TextNodes.
    BM25 matches on WORDS, so it needs this raw text — embeddings alone won't do."""
    data = chroma_collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or [{}] * len(documents)

    nodes = [
        TextNode(text=doc, metadata=meta or {})
        for doc, meta in zip(documents, metadatas)
        if doc  # skip any empty entries
    ]
    if not nodes:
        raise ValueError(
            "No text nodes found in Chroma. Did ingestion run and store documents? "
            "Check that chroma_db/ exists and the collection name matches."
        )
    print(f"  Loaded {len(nodes)} text nodes from Chroma for BM25.")
    return nodes


def build_hybrid_retriever_only(index, chroma_collection):
    """Semantic + BM25 fused retriever, without reranking or an LLM attached.
    Used by callers (like the chat engine in app2.py) that apply their own
    reranker/postprocessors on top."""
    # PIECE 1: semantic retriever (meaning)
    vector_retriever = index.as_retriever(similarity_top_k=RETRIEVER_K)

    # PIECE 2: BM25 keyword retriever (exact words) — fed raw text from Chroma
    nodes = get_nodes_from_chroma(chroma_collection)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=RETRIEVER_K,
    )

    # PIECE 3: fusion — run both, merge by rank (RRF)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        mode="reciprocal_rerank",
        num_queries=1,
        similarity_top_k=RETRIEVER_K,
        use_async=False,
    )
    return fusion_retriever


def build_hybrid_query_engine(index, chroma_collection):
    fusion_retriever = build_hybrid_retriever_only(index, chroma_collection)

    # PIECE 4: reranker — cross-encoder filters fused pool to top 4
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=RERANK_TOP_N,
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        node_postprocessors=[reranker],
        text_qa_template=QA_PROMPT,
    )
    return query_engine


def answer_question(query_engine, question: str):
    response = query_engine.query(question)
    sources = []
    for node in response.source_nodes:
        sources.append({
            "file": node.metadata.get("file_name", "unknown"),
            "score": round(node.score, 3) if node.score else None,
        })
    return str(response), sources


if __name__ == "__main__":
    index, chroma_collection = load_index_and_collection()
    query_engine = build_hybrid_query_engine(index, chroma_collection)

    print("\nNigeria Tax RAG (HYBRID + reranking) — ask a question ('quit' to exit)\n")

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
        print("\nSOURCES (after hybrid + rerank):")
        for s in sources:
            print(f"  - {s['file']} (score: {s['score']})")
        print("-" * 60 + "\n")