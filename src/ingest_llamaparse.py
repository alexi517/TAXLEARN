"""
ingest.py — Ingestion pipeline (LlamaParse + LlamaIndex) for the Nigeria Tax RAG project.

WHY LlamaParse: your tax PDFs are full of tables (rate schedules, income
thresholds, brackets). Basic PDF parsers flatten these into garbled text, which
wrecks retrieval — someone asks "what's the rate for income band X" and the
model gets a mangled chunk. LlamaParse converts tables into clean Markdown so
the structure survives into your vector store.

WHAT THIS DOES:
  1. Parses PDFs with LlamaParse -> clean Markdown (tables preserved)
  2. Splits the Markdown into chunks (~512 tokens, 50 overlap)
  3. Embeds each chunk with a local, free embedding model
  4. Stores chunks + vectors in a Chroma vector database on disk
  5. Runs a test retrieval so you can confirm it works

SETUP BEFORE RUNNING:
  1. Get a free LlamaParse API key at https://cloud.llamaindex.ai
  2. Set it as an environment variable:
       Mac/Linux:  export LLAMA_CLOUD_API_KEY="llx-..."
       Windows:    setx LLAMA_CLOUD_API_KEY "llx-..."
  3. pip install the extra package (see requirements note at bottom)

Run with:  python src/ingest.py
"""

import os
from pathlib import Path

import chromadb
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_parse import LlamaParse

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = r"C:\Users\NEW USER\Desktop\Rag_Application\Data"
CHROMA_DIR = r"C:\Users\NEW USER\Desktop\Rag_Application\Data\chroma_db"
COLLECTION_NAME = "nigeria_tax"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# LlamaParse reads this from the environment. Never hardcode API keys.
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY")


def parse_documents():
    """Parse all PDFs in DATA_DIR into Markdown using LlamaParse."""
    if not LLAMA_CLOUD_API_KEY:
        raise EnvironmentError(
            "LLAMA_CLOUD_API_KEY not set. Get a free key at "
            "https://cloud.llamaindex.ai and export it as an environment variable."
        )

    # result_type="markdown" is the key setting — it makes LlamaParse output
    # Markdown with tables preserved as Markdown tables.
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        # A parsing instruction nudges the parser for your document type.
        # Optional, but helps on dense legal/tax docs.
        parsing_instruction=(
            "This is a Nigerian tax law document. Preserve all tables, rate "
            "schedules, and numbered sections accurately as Markdown."
        ),
        verbose=True,
    )

    data_path = Path(DATA_DIR)
    pdf_files = list(data_path.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found under '{DATA_DIR}'.")

    print(f"Found {len(pdf_files)} PDF(s). Parsing with LlamaParse ...")
    all_documents = []
    for pdf in pdf_files:
        print(f"  Parsing {pdf.name} ...")
        # load_data returns Document objects containing the parsed Markdown.
        docs = parser.load_data(str(pdf))
        # attach the source filename so we can cite it later
        for d in docs:
            d.metadata["file_name"] = pdf.name
        all_documents.extend(docs)

    print(f"Parsed into {len(all_documents)} document object(s).")
    return all_documents


def build_index(documents):
    """Chunk, embed, and store the parsed documents in Chroma."""
    print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    # Because our documents are now Markdown, MarkdownNodeParser splits along
    # Markdown structure (headers, sections) which respects the document's
    # natural boundaries better than a plain sentence splitter.
    Settings.node_parser = MarkdownNodeParser()

    print(f"Setting up Chroma vector store at '{CHROMA_DIR}' ...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking, embedding, and indexing ...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("Index built and saved to disk.")
    return index


def test_query(index):
    """Run a sample retrieval to confirm it works."""
    print("\n" + "=" * 60)
    print("TEST QUERY — retrieving chunks (no LLM answer yet)")
    print("=" * 60)

    retriever = index.as_retriever(similarity_top_k=3)
    question = "What is the personal income tax rate structure?"
    print(f"\nQuestion: {question}\n")

    results = retriever.retrieve(question)
    for i, node in enumerate(results, 1):
        source = node.metadata.get("file_name", "unknown")
        score = node.score
        preview = node.text[:300].replace("\n", " ")
        print(f"--- Result {i} | source: {source} | score: {score:.3f} ---")
        print(f"{preview}...\n")


if __name__ == "__main__":
    docs = parse_documents()
    index = build_index(docs)
    test_query(index)
    print("\nDone. If you see relevant chunks (with intact tables) above, ingestion works.")
    print("Next: build query.py (retrieval + LLM generation with citations).")


# ---------------------------------------------------------------------------
# requirements.txt additions for this version:
#   llama-parse==0.5.19
#   llama-index-core==0.11.23
#   llama-index-embeddings-huggingface==0.3.1
#   llama-index-vector-stores-chroma==0.2.1
#   chromadb==0.5.15
#   sentence-transformers==3.2.1
# (llama-parse replaces the need for pypdf here, since LlamaParse does parsing)
# ---------------------------------------------------------------------------
