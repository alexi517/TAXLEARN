TaxLearn

Production-oriented RAG system for Nigeria's 2025 tax legislation.

TaxLearn is an end-to-end Retrieval-Augmented Generation (RAG) application that answers questions about Nigeria's 2025 tax laws using the source legislation as its knowledge base.

It combines hybrid retrieval, cross-encoder reranking, conversational query processing, grounding guardrails, and RAG evaluation to prioritize reliable answers over simple LLM responses.

💼 Project Highlights

- 🔎 Hybrid retrieval: Dense embeddings + BM25 + Reciprocal Rank Fusion
- 🎯 Reranking: Cross-encoder relevance scoring
- 📊 Evaluation: RAGAS evaluation with ~40 manually verified questions
- 📄 Document intelligence: Table-aware ingestion for legislation and tax schedules
- 🛡️ Grounding: Answers are constrained to retrieved source material
- 🚫 Hallucination control: Refuses questions outside the indexed knowledge base
- 💬 Conversational RAG: Query condensation for follow-up questions
- ⚙️ Backend: FastAPI API with interactive Swagger documentation
- 🖥️ Application: Streaming Streamlit interface
- 🐳 Deployment: Dockerized API

---

🏗️ Architecture

                    DOCUMENT INGESTION

Official Tax PDFs
       │
       ▼
  LlamaParse
       │
       ▼
Table-aware Markdown
       │
       ▼
    Chunking
       │
       ▼
BGE Embeddings
       │
       ▼
   ChromaDB


                    QUERY PIPELINE

User Question
       │
       ▼
Query Condensation
       │
       ├───────────────┐
       ▼               ▼
Dense Retrieval    BM25 Retrieval
       │               │
       └───────┬───────┘
               ▼
          RRF Fusion
               │
               ▼
      Cross-Encoder Rerank
               │
               ▼
      Grounded LLM Prompt
               │
               ▼
        Cited Answer

---

🔬 Key Engineering Decisions

Hybrid Retrieval

Legal documents contain both semantic concepts and exact terminology such as section numbers, tax names, and rates.

TaxLearn combines:

Dense retrieval for semantic similarity
+
BM25 for exact keyword matching
↓
Reciprocal Rank Fusion

This provides a broader retrieval signal than relying on a single retrieval strategy.

Cross-Encoder Reranking

Retrieved candidates are reranked using a cross-encoder before being passed to the LLM.

This separates:

Recall → Reranking → Precision

Table-Aware Ingestion

Tax legislation contains important tables and schedules.

Instead of relying on basic PDF text extraction, documents are parsed with LlamaParse and converted into structured Markdown before chunking so that table information remains usable during retrieval.

Conversational Retrieval

Follow-up questions such as:

«"What about companies?"»

can lack context when searched independently.

TaxLearn uses conversation history to transform follow-up questions into standalone retrieval queries.

Grounding & Refusal

The generation layer is instructed to use retrieved evidence and avoid unsupported claims.

Questions outside the indexed corpus are refused rather than answered from model knowledge.

---

📊 Evaluation

TaxLearn includes a RAGAS evaluation harness with approximately 40 manually verified questions covering factual, table-based, retrieval-sensitive, and out-of-scope queries.

Baseline — Dense Retrieval

Metric| Score
Faithfulness| 0.76
Answer Relevancy| 0.76
Context Recall| 0.69
Context Precision| 0.46

The baseline identified context precision as the main retrieval weakness.

That result drove the move from dense-only retrieval toward:

Dense Retrieval
      ↓
Evaluation
      ↓
Identify Retrieval Noise
      ↓
Hybrid Retrieval + RRF
      ↓
Cross-Encoder Reranking
      ↓
Re-evaluation

The improved pipeline is currently being evaluated. No unverified improvement figures are claimed.

«Evaluation note: approximately 71% of the baseline scoring calls completed because of free-tier LLM rate limits. Results should therefore be treated as directional rather than a definitive benchmark.»

---

🖥️ Application

The deployed application provides:

- Streaming AI responses
- Source-grounded answers
- Citations
- Conversational follow-ups
- Out-of-scope refusal
- FastAPI access

---

🛠️ Tech Stack

AI / Retrieval

"LlamaIndex" "LlamaParse" "BGE Embeddings" "ChromaDB" "BM25" "RAGAS" "Cross-Encoder"

Backend / Application

"Python" "FastAPI" "Streamlit"

LLM

"Groq" "Llama 3.3"

Infrastructure

"Docker" "Git"

---

🚀 Run Locally

git clone https://github.com/alexi517/TAXLEARN.git
cd TAXLEARN

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

Create ".env":

GROQ_API_KEY=your_key
LLAMA_CLOUD_API_KEY=your_key

Build the knowledge base:

python src/ingest.py

Run Streamlit:

streamlit run app2.py

Or run the API:

uvicorn src.api:app --reload

API documentation:

http://127.0.0.1:8000/docs

Docker

docker build -t taxlearn .
docker run -p 8000:8000 --env-file .env taxlearn

---

⚠️ Limitations

- Current corpus focuses on selected 2025 Nigerian tax reform legislation.
- It is not an exhaustive database of Nigerian tax law.
- New amendments and publications require re-ingestion.
- Public deployment runs on free-tier infrastructure and may experience cold-start latency.
- Conversations are currently session-based.
- The system should not be treated as professional tax or legal advice.

---

🔮 Future Improvements

- Complete before/after retrieval benchmark
- Passage-level citation highlighting
- Larger evaluation dataset
- Retrieval caching
- Persistent conversations
- User authentication
- Automated detection of updated legislation
- Expanded state-level tax coverage

---

🎓 What This Project Demonstrates

TaxLearn demonstrates an end-to-end AI engineering workflow:

Problem
  ↓
Data Ingestion
  ↓
Retrieval Architecture
  ↓
LLM Integration
  ↓
Evaluation
  ↓
Reliability Guardrails
  ↓
API + Application
  ↓
Docker Deployment

The project focuses on measuring and improving AI system behavior, rather than treating an LLM response as the finished product.

---

👨‍💻 Alex Moses

AI Engineer | Generative AI • RAG • AI Agents • Production Applications

"GitHub" (https://github.com/alexi517) · "LinkedIn" (https://linkedin.com/in/moses-alex)
