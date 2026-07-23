"""
run_eval.py — Evaluation harness using GROQ (free tier) for scoring.

Why Groq: RAGAS scoring makes many LLM judge-calls, which blew through the
Gemini free quota. Groq's free tier is more generous and very fast, so we use
it for the scoring step. Answer generation can stay on Gemini OR also use Groq —
this version uses Groq for BOTH to keep you off the Gemini ceiling entirely.

WHAT IT DOES (unchanged):
  1. Loads your eval set (eval/eval_set.json)
  2. Runs each question through your RAG system (answer + retrieved context)
  3. Scores with RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
  4. Prints your baseline metrics

SETUP:
  - GROQ_API_KEY in your .env  (free key from https://console.groq.com)
  - python -m pip install langchain-groq llama-index-llms-groq ragas datasets

Run with:  python eval/run_eval.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# make the project root importable so we can reuse src/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from llama_index.core import Settings, VectorStoreIndex  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.llms.groq import Groq as LlamaGroq  # noqa: E402
from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: E402
import chromadb  # noqa: E402

from datasets import Dataset  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.metrics import (  # noqa: E402
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "nigeria_tax"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # MUST match ingestion
GROQ_MODEL = "llama-3.1-8b-instant"            # llama-3.3-70b-versatile's 100k TPD free cap was exhausted; 8b has its own separate quota
TOP_K = 4
EVAL_SET_PATH = "eval/eval_set.json"
# ---------------------------------------------------------------------------


def load_index():
    """Reconnect to the existing Chroma store, using Groq as the LLM."""
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = LlamaGroq(model=GROQ_MODEL)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index


def collect_predictions(index, questions):
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    query_engine = index.as_query_engine(similarity_top_k=TOP_K)

    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"  [{i}/{len(questions)}] {q[:60]}...")
        response = query_engine.query(q)
        data["question"].append(q)
        data["answer"].append(str(response))
        data["contexts"].append([n.text for n in response.source_nodes])
        data["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(data)


def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    questions = eval_data["questions"]

    unfilled = [q for q in questions if "REPLACE ME" in q["ground_truth"]]
    if unfilled:
        print(f"WARNING: {len(unfilled)} questions still have placeholder answers. "
              f"Fill them in first or scores will be meaningless.\n")

    print(f"Loaded {len(questions)} eval questions.\n")

    print("Running RAG system on eval questions ...")
    index = load_index()
    dataset = collect_predictions(index, questions)

    print("\nScoring with RAGAS via Groq (fast, free) ...")
    # RAGAS needs a judge LLM + embeddings. Judge = Groq; embeddings = local (free).
    judge_llm = LangchainLLMWrapper(ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=2048))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    )

    # answer_relevancy samples 3 completions per question by default; Groq's
    # API rejects n>1 outright ('n' : number must be at most 1), so force 1.
    answer_relevancy.strictness = 1

    from ragas.run_config import RunConfig

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        # This model has a 6k tokens/minute free-tier cap; some jobs will
        # still fail even with retries. This config finished in ~52 minutes
        # with ~71% of jobs succeeding — the best time/reliability tradeoff
        # found. max_workers=1 + long retries (tried) made it worse: single
        # slow jobs and connection errors compounded into multi-hour runs.
        run_config=RunConfig(max_workers=4, max_retries=6, max_wait=60),
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (your baseline — write these down!)")
    print("=" * 60)
    print(result)
    print("=" * 60)
    print("\nAfter Stage 4 (hybrid retrieval + reranking), re-run this and")
    print("compare. The before/after is your headline CV metric.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# requirements.txt additions for this version:
#   ragas==0.2.10
#   datasets
#   langchain-groq
#   langchain-huggingface
#   llama-index-llms-groq
# ---------------------------------------------------------------------------