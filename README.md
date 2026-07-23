# TaxLearn — Production RAG with Evaluation

### Ask anything about Nigeria's 2025 tax laws — grounded, cited, clear.

A conversational retrieval-augmented generation (RAG) system that answers questions about Nigeria's 2025 tax laws in plain English, grounded in the official legislation with cited sources. Built with table-aware ingestion, hybrid retrieval, cross-encoder reranking, and a RAGAS evaluation harness — served through a streaming chat interface and a FastAPI backend.

**🔗 Live demo:** https://taxlearngit-czfepmz77agmutui54srz3.streamlit.app/
**📂 Code:** https://github.com/alexi517/TAXLEARN

---

## What it does

Nigeria overhauled its tax system in 2025 (the Nigeria Tax Act, Tax Administration Act, and related laws). TaxLearn lets anyone ask natural-language questions — *"Who is exempt from personal income tax?"*, *"What is the Tertiary Education Tax rate?"* — and returns answers grounded strictly in the official documents, with sources shown for every answer. Follow-up questions work conversationally. If a question falls outside the documents, it says so rather than hallucinating.

---

## Why it's built the way it is

This project prioritises **retrieval quality and evaluation rigour** over simply wiring up an LLM. The key engineering decisions:

- **Table-aware ingestion.** Tax documents are full of rate schedules and tables that standard PDF parsers garble. Ingestion uses LlamaParse to convert documents (tables intact) into clean Markdown, then splits on Markdown structure — so rate tables survive into retrieval rather than being cut mid-table.

- **Hybrid retrieval (semantic + keyword).** Dense embeddings are fuzzy on exact terms like section numbers and named taxes; BM25 keyword search misses paraphrase. TaxLearn runs both and fuses them with Reciprocal Rank Fusion, catching meaning-based *and* exact-term matches.

- **Cross-encoder reranking.** Initial retrieval casts a wide net for recall; a cross-encoder then re-scores candidates by true relevance and keeps only the best few for precision. This directly targets the noisy-retrieval weakness measured in baseline evaluation.

- **Conversational memory via query condensation.** Follow-ups like *"what about companies?"* are meaningless to retrieval alone. Before retrieving, the conversation history and new question are condensed into a standalone query — so follow-ups retrieve correctly.

- **Grounding guardrails.** The system prompt constrains answers to retrieved context and requires refusal when information isn't present — verified with out-of-scope test cases in the evaluation set.

---

## Architecture

```
                          INGESTION (run once)
  PDFs ──► LlamaParse ──► Markdown chunking ──► embed (BGE) ──► Chroma
        (tables preserved)

                          QUERY (per message)
  Question + history ──► condense to standalone query
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
  Dense retrieval (semantic)              BM25 retrieval (keyword)
        └─────────────────► RRF fusion ◄────────────┘
                              │
                     Cross-encoder rerank
                              │
                   Grounding prompt ──► LLM ──► streamed, cited answer

  Interfaces:  Streamlit chat (threads, streaming)  ·  FastAPI /query
```

---

## Evaluation

Retrieval and answer quality are measured with **RAGAS** against a hand-written evaluation set of ~40 questions (factual, table-based, and out-of-scope), with ground-truth answers verified against the source documents.

**Baseline — dense retrieval only:**

| Metric | Score |
|---|---|
| Faithfulness | 0.76 |
| Answer Relevancy | 0.76 |
| Context Precision | 0.46 |
| Context Recall | 0.69 |

The baseline shows a well-grounded system (high faithfulness) with noisy retrieval (low context precision) — which motivated the hybrid-retrieval and reranking work.

**After hybrid retrieval + reranking:** `measurement in progress`

> **Honest note on the baseline:** these figures come from a run in which ~71% of scoring calls completed (the remainder timed out on free-tier LLM rate limits), so they are directional rather than definitive. A full clean re-run on the improved system is pending and will replace this section with a complete before/after comparison.

---

## Tech stack

**Retrieval & generation:** LlamaIndex · LlamaParse · ChromaDB · BGE embeddings · BM25 · cross-encoder reranker · Groq (Llama 3.3)
**Evaluation:** RAGAS
**Serving:** Streamlit (chat UI) · FastAPI (API) · Docker
**Language:** Python

---

## Running locally

```bash
# 1. clone
git clone https://github.com/alexi517/TAXLEARN.git && cd TAXLEARN

# 2. environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

# 3. install
pip install -r requirements.txt

# 4. add keys to a .env file (see .env.example)
#    GROQ_API_KEY=...
#    LLAMA_CLOUD_API_KEY=...   (ingestion only)

# 5. ingest documents (builds chroma_db/)
python src/ingest.py

# 6a. chat interface
streamlit run app2.py

# 6b. or the API
uvicorn src.api:app --reload      # docs at http://127.0.0.1:8000/docs
```

**Docker (API):**
```bash
docker build -t taxlearn .
docker run -p 8000:8000 --env-file .env taxlearn
```

> **Data:** source documents are not committed. The corpus is Nigeria's 2025 tax reform legislation (Nigeria Tax Act, Tax Administration Act, Nigeria Revenue Service Act, Joint Revenue Board Act) plus the explanatory memorandum, available from official sources including nrs.gov.ng.

---

## Limitations & next steps

- **Corpus scope:** covers the 2025 tax reform acts and key supporting documents; not exhaustive of all Nigerian tax law or later amendments.
- **Agency transition:** documents reflect the FIRS → Nigeria Revenue Service (NRS) rename; ingesting newer NRS-era publications is future work.
- **Evaluation:** a full clean before/after RAGAS run on the improved system is pending (see note above).
- **Latency:** the deployed demo runs on free-tier CPU, so the first query after idle is slow (cold start plus reranker warm-up). Retrieval breadth and reranker size were tuned for latency; the quality cost of that tuning is to be verified against the evaluation set.
- **Persistence:** conversation threads are session-scoped; there is no cross-session storage or user accounts.
- **Possible extensions:** passage-level citation highlighting, expanding to state-level tax rules, caching frequent queries.

---

## What I learned building this

Retrieval quality — not the language model — is where most RAG systems succeed or fail. Building the evaluation harness turned a vague sense that "it seems to work" into a concrete, measurable weakness (context precision of 0.46), which pointed directly at the fix. Every subsequent improvement decision — hybrid retrieval, reranking, and later latency tuning — became a data-driven trade-off rather than guesswork.
