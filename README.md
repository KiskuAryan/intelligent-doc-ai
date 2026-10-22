# ⚡ DocAI — Local & Private Intelligent Document RAG

> High-performance, offline Retrieval-Augmented Generation (RAG) system running entirely on Apple Silicon GPU via Apple MLX.

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=flat-square&logo=python)
![Apple Silicon](https://img.shields.io/badge/Accelerated-Apple%20Metal%20%28MLX%29-9945FF?style=flat-square&logo=apple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local%20%26%20Offline-10B981?style=flat-square)
[![Tests](https://img.shields.io/badge/Tests-16%20Passing-brightgreen.svg)](#-automated-test-suite)

---

## 🌟 Highlights

- **🔒 100% Local & Private**: No cloud APIs, no telemetry, zero internet connection required. Your documents never leave your device.
- **⚡ Hardware Accelerated (MLX)**: Powered by Apple’s native **MLX** framework running in unified memory on Apple Silicon (M1/M2/M3/M4/M5).
- **🚀 50+ Tokens/Sec**: Employs a 4-bit quantized `Phi-3-mini-4k-instruct` (~2.2 GB memory footprint) generating answers fluidly in real-time.
- **⚡ Real-time Token Streaming**: Word-by-word streaming with sub-second time-to-first-token.
- **🎯 Grounded with Citations**: Automatic source attribution with exact page numbers and cosine similarity confidence scores.
- **⚡ Parallel Batch Indexing**: Large multi-page documents and books are parsed and indexed in seconds using GPU-batched embeddings (`bge-small-en-v1.5`).

---

## 🏗️ Architecture

```
[ Upload PDF / Document ]
            │
            ▼
   [ PyMuPDF Parser ]        ── Page-level text extraction
            │
            ▼
    [ Text Sanitizer ]       ── Token cleaning & line normalization
            │
            ▼
    [ Sliding Chunker ]      ── 300 words with 50-word overlap
            │
            ▼
[ GPU Batch Embedder ]       ── BAAI/bge-small-en-v1.5 (384-dim L2)
            │
            ▼
   [ FAISS Vector DB ]       ── Cosine Similarity (IndexFlatIP)
            │
    ┌───────┴────────────────────────┐
    ▼                                ▼
[ User Query ]              [ Quick Summary ]
    │                                │
    ▼                                ▼
[ Top-K Context Match ] ─────► [ Strict Grounding Prompt ]
                                     │
                                     ▼
                           [ Apple MLX Generator ]
                        (Phi-3-mini 4-bit on Metal GPU)
                                     │
                                     ▼
                          [ Real-time Token Stream ]
                           + Verified Page Citations
```

---

## 🚀 Quickstart Guide

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.11 or Python 3.12 (Recommended: `brew install python@3.12`)

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd intelligent-doc-ai

# Create virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application

#### Interactive Web UI (Recommended)
```bash
streamlit run app.py
```
*Opens automatically at `http://localhost:8501`.*

#### Command-Line Pipeline (CLI)
```bash
python main.py
```
*Executes an end-to-end extraction, indexing, and question-answering benchmark on `data/raw/sample.pdf`.*

---

## 📂 Project Structure

```
intelligent-doc-ai/
├── app.py                  # Streamlit web application with modern glassmorphism UI
├── main.py                 # CLI end-to-end pipeline runner
├── requirements.txt        # Python package dependencies
├── .gitignore              # Git ignore rules for virtualenvs, caches & data
├── data/
│   └── raw/
│       └── sample.pdf      # Sample research paper (Attention Is All You Need)
├── utils/
│   ├── pdf_parser.py       # Robust PyMuPDF extractor (bytes & file path)
│   ├── cleaner.py          # Non-destructive text sanitization
│   ├── chunker.py          # Sliding-window text chunker with overlap
│   ├── embedder.py         # SentenceTransformers with GPU batch encoding
│   ├── retriever.py        # FAISS vector database & cosine similarity search
│   └── generator.py        # Apple MLX LLM loader & streaming generator
└── tests/
    ├── test_cleaner.py     # Unit tests for text cleaning and regex sanitization
    ├── test_chunker.py     # Unit tests for chunk window math and boundary checks
    ├── test_retriever.py   # Unit tests for FAISS index creation and cosine similarity
    └── test_pdf_parser.py  # Unit tests for PyMuPDF in-memory PDF extraction
```

---

## 🧪 Automated Pytest Suite

DocAI includes automated unit and boundary tests covering document ingestion, text sanitization, chunking mathematics, FAISS vector indexing, and cosine retrieval:

```bash
pytest tests/ -v
```

```text
tests/test_chunker.py::test_chunk_text_empty PASSED                      [  6%]
tests/test_chunker.py::test_chunk_text_smaller_than_chunk_size PASSED    [ 12%]
tests/test_chunker.py::test_chunk_text_sliding_window_overlap PASSED     [ 18%]
tests/test_chunker.py::test_chunk_text_invalid_parameters PASSED         [ 25%]
tests/test_cleaner.py::test_clean_text_empty_and_none PASSED             [ 31%]
tests/test_cleaner.py::test_clean_text_removes_special_tokens PASSED     [ 37%]
tests/test_cleaner.py::test_clean_text_removes_chat_markers PASSED       [ 43%]
tests/test_cleaner.py::test_clean_text_removes_isolated_page_numbers PASSED [ 50%]
tests/test_cleaner.py::test_clean_text_deduplicates_consecutive_lines PASSED [ 56%]
tests/test_cleaner.py::test_clean_text_normalizes_whitespace PASSED      [ 62%]
tests/test_pdf_parser.py::test_extract_text_from_pdf_bytes PASSED        [ 68%]
tests/test_pdf_parser.py::test_extract_text_from_multi_page_pdf PASSED   [ 75%]
tests/test_retriever.py::test_create_faiss_index_basic PASSED            [ 81%]
tests/test_retriever.py::test_create_faiss_index_empty PASSED            [ 87%]
tests/test_retriever.py::test_search_similar_chunks PASSED               [ 93%]
tests/test_retriever.py::test_search_similar_chunks_fewer_than_top_k PASSED [100%]

======================== 16 passed in 0.45s ========================
```

---

## ⚙️ Tech Stack & Models

| Component | Technology | Details |
| :--- | :--- | :--- |
| **LLM** | `mlx-community/Phi-3-mini-4k-instruct-4bit` | 3.8B params, 4-bit quantized, ~2.2 GB RAM |
| **Inference** | Apple MLX (`mlx-lm`) | Native Apple Silicon Metal acceleration |
| **Embeddings**| `BAAI/bge-small-en-v1.5` | 384-dimensional dense vectors ($L_2$ normalized) |
| **Vector Index** | FAISS (`IndexFlatIP`) | Exact Cosine Similarity search |
| **PDF Parser** | PyMuPDF (`pymupdf`) | Fast in-memory stream & page extraction |
| **Testing** | Pytest (`pytest>=8.0.0`) | Automated test suite (16 passing tests) |
| **Frontend** | Streamlit | Minimalist dark theme with CSS custom styling |

---

## 📄 License
MIT License. Free for personal and commercial use.
