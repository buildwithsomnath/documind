# 🧠 DocuMind — Enterprise RAG Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)
![pgvector](https://img.shields.io/badge/pgvector-HNSW_0.7+-green.svg)
![Gemini](https://img.shields.io/badge/Google_Gemini-Flash_API-orange.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![Vite](https://img.shields.io/badge/Vite-6.x-purple.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

### Grounded Document Intelligence with Hybrid Retrieval, Cross-Encoder Reranking & Verifiable Citations

A production-grade, enterprise Retrieval-Augmented Generation (RAG) platform where users securely ingest unstructured documents (PDF, DOCX), execute hybrid semantic and lexical search, and receive strictly factual, hallucination-resistant answers grounded with interactive passage citations.

</div>

---

## 📖 Overview

**DocuMind** is a full-stack, enterprise-ready Retrieval-Augmented Generation platform built to solve the core challenges of enterprise knowledge search: **retrieval recall**, **hallucination prevention**, and **provenance verification**.

Standard RAG architectures often suffer from vector search blind spots (e.g. specific acronyms, code identifiers, or technical IDs) and generate unverified answers. DocuMind addresses this with a **Two-Stage Hybrid Search & Reranking Architecture**:

1. **First-Stage Hybrid Retrieval:** Merges dense vector semantic search (`sentence-transformers/all-MiniLM-L6-v2` with pgvector HNSW indexing) and sparse lexical search (PostgreSQL Full-Text Search with `websearch_to_tsquery` and cover-density ranking) via **Reciprocal Rank Fusion (RRF, $k=60$)**.
2. **Second-Stage Cross-Encoder Reranking:** Deep contextual passage reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` with calibrated sigmoid relevance scoring.
3. **Grounded Generation & Citation Verification:** Synthesizes strictly grounded answers via Google Gemini, enforcing bracketed source provenance and extracting verified citations that link directly to an interactive passage drawer.
4. **Built-in RAG Evaluation Suite:** Evaluates retrieval quality (`Recall@K`, `MRR`, `nDCG@K`) and generation quality (`Faithfulness`, `Citation Accuracy`) across synthetic benchmark datasets.

This project is designed for:

- **AI & ML Engineers** building production-grade RAG architectures beyond naive LangChain demos.
- **Enterprise Engineering Teams** needing verifiable document intelligence over confidential documentation.
- **Researchers & Developers** experimenting with Hybrid Search, Cross-Encoders, and automated RAG evaluation.

---

# ✨ Features

## 📄 Document Ingestion & Page-Aware Parsing

- **Multi-Format Extraction:** High-fidelity text extraction from multi-page PDFs (`PyMuPDF`) and Word documents (`python-docx`).
- **Page-Aware Boundary Tracking:** Automatically maps every character to its original document page numbers (`page_start` to `page_end`).
- **Sliding-Window Chunking:** Configurable 500-character chunks with 100-character overlap, tuned specifically for 256-token transformer embedding limits.
- **Duplicate Prevention:** Cryptographic SHA-256 document hashing avoids duplicate ingestions and prevents database bloat.
- **Dense Embeddings:** Local embedding generation via `all-MiniLM-L6-v2` (384-dimensional normalized vectors).

## 🔍 Two-Stage Hybrid Retrieval

- **Dense Semantic Search:** PostgreSQL `pgvector` HNSW index with cosine distance (`<=>`), optimized for conceptual questions.
- **Lexical Full-Text Search:** Native PostgreSQL `tsvector` with English stemming, `websearch_to_tsquery` query parsing, and `ts_rank_cd` cover-density ranking for keywords and technical identifiers.
- **Reciprocal Rank Fusion (RRF):** Merges vector and keyword candidate rankings using $RRF(d) = \sum \frac{1}{k + r(d)}$ ($k=60$) to eliminate parameter-tuning biases.
- **Cross-Encoder Reranking:** Second-stage deep contextual cross-attention (`ms-marco-MiniLM-L-6-v2`) scoring the top passages with sigmoid calibration.

## 🤖 Grounded Generation & Anti-Hallucination

- **Strict Grounding Prompts:** Enforces zero outside speculation; requires explicit negative trigger (`"I do not have enough information in the provided documents to answer this question."`) when context is insufficient.
- **Resilient Gemini Client:** Direct REST client supporting `gemini-flash-latest`, `gemini-2.5-flash`, and `gemini-2.5-flash-lite` with exponential backoff retry and automatic model fallback on HTTP 503 / 429 / socket timeouts.

## 🏷️ Verifiable Citations & Passage Drawer

- **Automatic Citation Validation:** Parses bracketed inline citations (`[filename.pdf, Page X]`), verifies claims against retrieved passage metadata, and calculates `citation_accuracy_percentage`.
- **Interactive Deep-Dive UI:** Clicking any citation pill opens an animated slide drawer displaying the exact retrieved chunk, relevance score, retrieval pipeline path, and source metadata.

## 📊 Built-In RAG Evaluation Suite

- **Automated Retrieval Benchmarks:** Measures `Recall@K`, `Mean Reciprocal Rank (MRR)`, and `Normalized Discounted Cumulative Gain (nDCG@3)`.
- **Generation Quality Checks:** Calculates factual faithfulness scores and tests anti-hallucination guardrails against unanswerable test queries.
- **Interactive Dashboard:** Visual comparison cards in the React frontend comparing Vector vs. Keyword vs. Hybrid Reranked search.

## 🐳 Containerized Deployment

- **Multi-Container Docker Compose:** Production setup orchestrating `pgvector` PostgreSQL 16, Gunicorn-powered Flask API, and Nginx-powered React SPA.
- **Zero-Wait Pre-Cached Models:** HuggingFace transformer models pre-cached during Docker build for instant container startup.

---

# 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        React Frontend (Vite)                           │
│  - Document Ingestion   - Interactive Chat   - Slide Passage Drawer   │
│  - Document List        - Citation Cards     - Evaluation Dashboard   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST (/api)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Flask REST API (Gunicorn)                          │
│                                                                        │
│  ┌────────────────────┐   ┌────────────────────┐   ┌────────────────┐ │
│  │ Document Extractor │──►│ Chunking Engine    │──►│ Embedder       │ │
│  │ (PyMuPDF / DOCX)   │   │ (500 char/100 ovlp)│   │ (all-MiniLM)   │ │
│  └────────────────────┘   └────────────────────┘   └───────┬────────┘ │
│                                                            │          │
│  ┌─────────────────────────────────────────────────────────┼────────┐ │
│  │ Retrieval & Reranking Engine                            │          │
│  │                                                         ▼          │
│  │  ┌───────────────┐     ┌───────────────┐   ┌────────────────────┐  │
│  │  │ Vector Search │     │ Keyword Search│   │ PostgreSQL 16      │  │
│  │  │ (pgvector <=>)│     │ (ts_rank_cd)  │   │ + pgvector (HNSW)  │  │
│  │  └───────┬───────┘     └───────┬───────┘   └────────────────────┘  │
│  │          └───────────┬─────────┘                                   │
│  │                      ▼                                             │
│  │          ┌───────────────────────┐                                 │
│  │          │ Reciprocal Rank Fusion│                                 │
│  │          └───────────┬───────────┘                                 │
│  │                      ▼                                             │
│  │          ┌───────────────────────┐                                 │
│  │          │ Cross-Encoder Reranker│ (ms-marco-MiniLM-L-6-v2)        │
│  │          └───────────┬───────────┘                                 │
│  └──────────────────────┼─────────────────────────────────────────────┘
│                         ▼
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ Grounded Generation Service                                      │ │
│  │  - Context Injection    - Prompt Grounding Rules                 │ │
│  │  - Google Gemini REST   - Automatic Model Fallback & Retry       │ │
│  │  - Citation Extraction  - Provenance Verification                │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ REST API
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Google Gemini API (Cloud)                           │
│              (gemini-flash-latest / gemini-2.5-flash)                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

# 📸 Screenshots

> Add your screenshots to the `screenshots/` folder and update the paths below.

## Knowledge Base & Document Ingestion

![Knowledge Base Screenshot](./screenshots/KnowledgeBase.png)

## Grounded Chat with Verified Citations

![Chat Interface Screenshot](./screenshots/ChatInterface.png)

## Interactive Passage Slide Drawer

![Passage Drawer Screenshot](./screenshots/PassageDrawer.png)

## RAG Evaluation & Benchmarking Dashboard

![Evaluation Dashboard Screenshot](./screenshots/EvaluationDashboard.png)

---

# 📂 Project Structure

```text
documind/
│
├── backend/
│   ├── app/
│   │   ├── generation/              # Grounded generation & citations
│   │   │   ├── citations.py         # Citation extraction & verification
│   │   │   ├── gemini_client.py     # Gemini REST client with retry & fallback
│   │   │   ├── prompt.py            # Grounded system prompts & context formatting
│   │   │   └── service.py           # End-to-end Q&A orchestration service
│   │   │
│   │   ├── ingestion/               # Document ingestion & embeddings
│   │   │   ├── chunker.py           # Page-aware sliding-window chunker
│   │   │   ├── embedder.py          # SentenceTransformer singleton service
│   │   │   ├── extractor.py         # PDF & DOCX text parsers
│   │   │   └── pipeline.py          # Complete ingestion pipeline
│   │   │
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   │   ├── chunk.py             # DocumentChunk model (Vector 384 + tsvector)
│   │   │   └── document.py          # Document metadata model
│   │   │
│   │   ├── retrieval/               # Two-stage hybrid search engine
│   │   │   ├── fusion.py            # Reciprocal Rank Fusion (RRF) math
│   │   │   ├── hybrid_search.py     # Hybrid search coordinator
│   │   │   ├── keyword_search.py    # PostgreSQL Full-Text Search
│   │   │   ├── reranker.py          # Cross-Encoder reranker service
│   │   │   ├── types.py             # SearchResult data structures
│   │   │   └── vector_search.py     # pgvector cosine similarity engine
│   │   │
│   │   ├── routes/                  # Flask REST API Blueprints
│   │   │   ├── chat.py              # POST /api/chat/ask
│   │   │   ├── documents.py         # Ingestion & document management
│   │   │   ├── evaluation.py        # POST /api/evaluation/run
│   │   │   └── search.py            # Vector, Keyword & Hybrid search endpoints
│   │   │
│   │   ├── config.py                # Centralized environment configuration
│   │   ├── db.py                    # SQLAlchemy database engine & sessions
│   │   └── __init__.py              # Flask app factory
│   │
│   ├── tests/                       # Pytest test suite (32 tests)
│   │   ├── test_evaluation.py       # Metrics & evaluator tests
│   │   ├── test_generation.py       # Prompt, citations & client tests
│   │   ├── test_ingestion.py        # Extractor, chunker & embedder tests
│   │   └── test_retrieval.py        # Vector, FTS, RRF & reranker tests
│   │
│   ├── Dockerfile                   # Backend Docker container specification
│   ├── manage.py                    # CLI management script (run, init-db, test, evaluate)
│   ├── pytest.ini                   # Pytest discovery configuration
│   └── requirements.txt             # Python dependencies
│
├── evaluation/                      # Offline RAG evaluation benchmark
│   ├── benchmark_dataset.json       # Golden Q&A test cases with negative queries
│   ├── evaluator.py                 # Multi-method evaluation engine
│   ├── metrics.py                   # Recall@K, MRR, nDCG@K & Faithfulness
│   └── run_evaluation.py            # Standalone benchmark CLI runner
│
├── frontend/                        # React + Vite application
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx    # Chat UI with citation pills
│   │   │   ├── DocumentList.jsx     # Document table with deletion
│   │   │   ├── DocumentUpload.jsx   # Drag-and-drop file upload
│   │   │   ├── EvaluationDashboard.jsx # RAG benchmark visualization
│   │   │   ├── Navbar.jsx           # Top navigation bar
│   │   │   └── PassageDrawer.jsx    # Slide-over source passage deep-dive
│   │   │
│   │   ├── services/
│   │   │   └── api.js               # Centralized frontend API client
│   │   │
│   │   ├── App.jsx                  # Main application layout
│   │   ├── index.css                # Custom styling & animations
│   │   └── main.jsx                 # React root mount
│   │
│   ├── Dockerfile                   # Frontend multi-stage Nginx build
│   ├── nginx.conf                   # Nginx reverse proxy configuration
│   ├── package.json                 # Node dependencies
│   └── vite.config.js               # Vite build & proxy settings
│
├── sample_docs/                     # Test documents for initial ingestion
│   ├── sample_architecture.pdf      # Multi-page system architecture PDF
│   └── sample_spec.docx             # Product requirements DOCX
│
├── screenshots/                     # README screenshot assets
├── .dockerignore                    # Docker build exclusion rules
├── .env                             # Environment variables & secrets
├── .gitignore                       # Git ignored files & directories
├── docker-compose.yml               # Multi-container orchestration
└── README.md                        # Documentation
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/buildwithsomnath/DocuMind.git

cd DocuMind/documind
```

---

# Option A: Full Multi-Container Docker Setup (Recommended)

Run the entire platform (PostgreSQL pgvector, Flask backend, and React Nginx frontend) with one command:

### 1. Configure Environment Variables
Create or verify your `.env` file in `documind/`:

```env
POSTGRES_PASSWORD=documind
DATABASE_URL=postgresql://documind:documind@localhost:5432/documind
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-latest
LLM_REQUEST_TIMEOUT=60
```

### 2. Launch with Docker Compose

```bash
docker compose up --build -d
```

### 3. Access Applications

- **Frontend UI:** `http://localhost:3000`
- **Backend API:** `http://localhost:5000/api/health`
- **PostgreSQL Database:** `localhost:5432`

To shut down:
```bash
docker compose down
```

---

# Option B: Local Development Setup

## 1. Start PostgreSQL with pgvector

Ensure Docker Desktop is running, then start the database container:

```bash
docker compose up -d db
```

---

## 2. Backend Setup

### Create Virtual Environment

#### Windows (PowerShell)

```powershell
cd backend
python -m venv ..\.venv
..\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
cd backend
python3 -m venv ../.venv
source ../.venv/bin/activate
```

---

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

### Configure `.env`

Create `.env` in `documind/.env`:

```env
POSTGRES_PASSWORD=documind
DATABASE_URL=postgresql://documind:documind@localhost:5432/documind
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-flash-latest
LLM_REQUEST_TIMEOUT=60
```

---

### Initialize Database Schema

Creates the `pgvector` extension, creates tables, and builds HNSW / GiST indexes:

```powershell
python manage.py init-db
```

---

### Run Automated Tests

Execute all 32 unit and integration tests:

```powershell
python manage.py test
```

---

### Start Backend Development Server

```powershell
python manage.py run
```

Backend URL:
```text
http://127.0.0.1:5000
```

---

## 3. Frontend Setup

In a separate terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:
```text
http://localhost:3000
```

---

# 🔌 API Endpoints

## Health & System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service health status & database connectivity |

## Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/documents` | List all ingested documents with metadata |
| POST | `/api/documents/upload` | Upload & ingest document (`multipart/form-data`) |
| GET | `/api/documents/<id>` | Document details & chunks (`?include_chunks=true`) |
| DELETE | `/api/documents/<id>` | Delete document and cascade chunks |

## Search & Retrieval

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search/vector` | Dense semantic vector search (cosine distance) |
| POST | `/api/search/keyword` | Lexical full-text search (`ts_rank_cd`) |
| POST | `/api/search/hybrid` | Hybrid RRF search with Cross-Encoder reranking |

## Grounded Chat & Q&A

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/ask` | Grounded Q&A with verifiable inline citations |

### Request Payload Example:
```json
{
  "question": "What is the chunk size and overlap configured in DocuMind?",
  "top_k": 3,
  "enable_rerank": true
}
```

### Response Payload Example:
```json
{
  "question": "What is the chunk size and overlap configured in DocuMind?",
  "answer": "DocuMind uses a chunk size of 500 characters paired with a 100-character overlap [sample_spec.docx, Page 1].",
  "total_citations": 1,
  "verified_citations": 1,
  "citation_accuracy_percentage": 100.0,
  "citations": [
    {
      "citation_text": "[sample_spec.docx, Page 1]",
      "filename": "sample_spec.docx",
      "page_number": 1,
      "verified": true,
      "chunk_id": "8b51dca9-411a-4ea9-bca1-f09b552fae7a",
      "score": 0.942
    }
  ],
  "sources_used": ["sample_spec.docx"],
  "retrieved_passages": [...]
}
```

## Evaluation Benchmark

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/evaluation/run` | Execute automated RAG benchmark evaluation |

---

# 🛠️ Tech Stack

## Backend & API

- **Python 3.12**
- **Flask 3.1** (RESTful API Blueprints)
- **SQLAlchemy 2.0** (ORM)
- **Psycopg 3** (High-performance PostgreSQL adapter)
- **Gunicorn** (Production WSGI Server)

## AI / ML & Embeddings

- **Sentence-Transformers** (`sentence-transformers/all-MiniLM-L6-v2`)
- **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **PyTorch**
- **Google Gemini API** (`gemini-flash-latest`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`)

## Database & Vector Storage

- **PostgreSQL 16**
- **pgvector 0.7+** (HNSW Vector Indexing, Cosine Distance)
- **PostgreSQL Full-Text Search** (`tsvector`, `tsquery`, `ts_rank_cd`)

## Document Parsing

- **PyMuPDF (fitz)** — Page-aware PDF parsing
- **python-docx** — Structured Word document parsing

## Frontend

- **React 18**
- **Vite 6**
- **Lucide Icons**
- **Modern CSS & Responsive Layouts**
- **Fetch API Client**

## DevOps & Containerization

- **Docker**
- **Docker Compose**
- **Nginx** (Alpine reverse proxy)

---

# 🧪 Automated Testing & Evaluation

### Running Unit & Integration Tests

DocuMind maintains comprehensive test coverage across ingestion, retrieval, reranking, prompt formatting, citation verification, and API endpoints:

```powershell
cd documind/backend
python manage.py test
```

**Results:**
```text
============================= 32 passed in 11.63s =============================
```

### Running RAG Benchmark Evaluation

Run the automated evaluation CLI comparing Vector vs. Keyword vs. Hybrid Reranked:

```powershell
python manage.py evaluate
```

Or trigger from the web interface at `http://localhost:3000` on the **"System Evaluation"** tab.

**Metrics Evaluated:**
- **Recall@K:** Measures the proportion of relevant ground-truth passages retrieved in top $K$.
- **Mean Reciprocal Rank (MRR):** Measures how high the first relevant document is placed in rankings.
- **Normalized Discounted Cumulative Gain (nDCG@K):** Evaluates ranking quality taking graded position into account.
- **Faithfulness Score:** Quantifies adherence to reference context.
- **Anti-Hallucination Guardrail Check:** Validates that unanswerable queries trigger the fallback rejection response.

---

# 🔮 Future Improvements

- [ ] **Multi-Modal Document Parsing:** Extracting tables, diagrams, and figures using OCR (PaddleOCR / Tesseract).
- [ ] **Conversational Memory:** Multi-turn session persistence with contextual query reformulation.
- [ ] **Semantic Caching:** Redis vector cache to serve frequent semantic queries with near-zero latency.
- [ ] **Query Expansion & HyDE:** Generating Hypothetical Document Embeddings for abstract queries.
- [ ] **Role-Based Access Control (RBAC):** Document permissions and multi-tenant organizational spaces.
- [ ] **Streaming Responses:** Token streaming via Server-Sent Events (SSE) or WebSockets.
- [ ] **Cloud Deployment:** Production Helm charts for Kubernetes and AWS ECS deployment.

---

# 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit Your Changes**
   ```bash
   git commit -m "feat: add amazing feature"
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

---

# 👨‍💻 Author

### Somnath Das

- **GitHub:** [@buildwithsomnath](https://github.com/buildwithsomnath)
- **Project Repository:** [DocuMind](https://github.com/buildwithsomnath/DocuMind)

---

# 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

# ⭐ Support

If you found DocuMind helpful in learning or building RAG platforms:

- ⭐ **Star the repository**
- 🍴 **Fork the project**
- 📢 **Share with fellow AI engineers and developers**

---

<div align="center">

Made with ❤️ by **Somnath Das**

</div>
