# ⚡ DocAI — Local & Private Intelligent Document RAG

> High-performance, offline Retrieval-Augmented Generation (RAG) system running entirely on Apple Silicon GPU via Apple MLX.

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=flat-square&logo=python)
![Apple Silicon](https://img.shields.io/badge/Accelerated-Apple%20Metal%20%28MLX%29-9945FF?style=flat-square&logo=apple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local%20%26%20Offline-10B981?style=flat-square)

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
└── utils/
    ├── pdf_parser.py       # Robust PyMuPDF extractor (bytes & file path)
    ├── cleaner.py          # Non-destructive text sanitization
    ├── chunker.py          # Sliding-window text chunker
    ├── embedder.py         # SentenceTransformers with GPU batch encoding
    ├── retriever.py        # FAISS vector database & cosine similarity search
    └── generator.py        # Apple MLX LLM loader & streaming generator
```

---

## ⚙️ Tech Stack & Models

| Component | Technology | Details |
| :--- | :--- | :--- |
| **LLM** | `mlx-community/Phi-3-mini-4k-instruct-4bit` | 3.8B params, 4-bit quantized, ~2.2 GB RAM |
| **Inference** | Apple MLX (`mlx-lm`) | Native Apple Silicon Metal acceleration |
| **Embeddings**| `BAAI/bge-small-en-v1.5` | 384-dimensional dense vectors ($L_2$ normalized) |
| **Vector Index** | FAISS (`IndexFlatIP`) | Exact Cosine Similarity search |
| **PDF Parser** | PyMuPDF (`fitz`) | Fast in-memory stream & page extraction |
| **Frontend** | Streamlit | Minimalist dark theme with CSS custom styling |

---

## 📄 License
MIT License. Free for personal and commercial use.
