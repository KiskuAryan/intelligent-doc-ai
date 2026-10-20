import streamlit as st
import os
import hashlib
from utils.pdf_parser import extract_text_from_pdf
from utils.cleaner import clean_text
from utils.chunker import chunk_text
from utils.embedder import generate_embedding, generate_embeddings
from utils.retriever import create_faiss_index, search_similar_chunks
from utils.generator import stream_answer

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DocAI — Private Document Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# BESPOKE MINIMAL & CLEAN STYLING (CSS)
# ==========================================
st.markdown("""
<style>
    /* Global Font & Theme */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Top Header Badges */
    .badge-container {
        display: flex;
        gap: 0.6rem;
        align-items: center;
        flex-wrap: wrap;
        margin-top: 0.4rem;
        margin-bottom: 1.5rem;
    }
    .badge {
        font-size: 0.76rem;
        font-weight: 500;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        letter-spacing: 0.01em;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .badge-purple {
        background: rgba(139, 92, 246, 0.12);
        color: #a78bfa;
        border: 1px solid rgba(139, 92, 246, 0.25);
    }
    .badge-slate {
        background: rgba(148, 163, 184, 0.1);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    /* Document Status Card */
    .doc-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .doc-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .doc-meta {
        font-size: 0.85rem;
        color: #94a3b8;
    }

    /* Chat Messages */
    .user-bubble {
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.25);
        color: #f1f5f9;
        padding: 0.85rem 1.15rem;
        border-radius: 12px 12px 2px 12px;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 85%;
        margin-left: auto;
    }
    .assistant-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #e2e8f0;
        padding: 1.15rem 1.35rem;
        border-radius: 12px 12px 12px 2px;
        margin-bottom: 1.25rem;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Citation Card */
    .citation-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #6366f1;
        padding: 0.65rem 0.9rem;
        margin-top: 0.5rem;
        font-size: 0.82rem;
        color: #cbd5e1;
        border-radius: 0 8px 8px 0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Metric Card in Sidebar */
    .sidebar-metric {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    .sidebar-metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .sidebar-metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
    }

    /* Button Styling */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.2s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    div.stButton > button:hover {
        border-color: #6366f1;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.2);
    }

    /* Hide standard Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document" not in st.session_state:
    st.session_state.document = None  # Dict: {name, pages, all_chunks, index, hash}
if "summary" not in st.session_state:
    st.session_state.summary = None

# ==========================================
# DOCUMENT INDEXING FUNCTION
# ==========================================
def index_document(pdf_bytes, file_name):
    """
    Parses PDF bytes, cleans text, chunks, batch-embeds, and creates a FAISS index.
    Saves state in st.session_state.
    """
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Avoid re-indexing if already indexed
    if st.session_state.document and st.session_state.document.get("hash") == file_hash:
        return st.session_state.document

    pages = extract_text_from_pdf(pdf_bytes)
    if not pages:
        return None

    all_chunks = []
    chunk_texts = []

    for page in pages:
        cleaned = clean_text(page["text"])
        chunks = chunk_text(cleaned, chunk_size=300, overlap=50)

        for idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 20:
                continue

            chunk_texts.append(chunk)
            all_chunks.append({
                "page": page["page"],
                "chunk_id": idx,
                "text": chunk
            })

    if not chunk_texts:
        return None

    # GPU-accelerated parallel batch embedding
    embeddings = generate_embeddings(chunk_texts, batch_size=32)
    index = create_faiss_index(embeddings)

    doc_data = {
        "name": file_name,
        "total_pages": len(pages),
        "all_chunks": all_chunks,
        "index": index,
        "hash": file_hash
    }

    st.session_state.document = doc_data
    st.session_state.messages = []
    st.session_state.summary = None
    return doc_data

# ==========================================
# PROMPT BUILDER
# ==========================================
def build_prompt(context, question):
    return f"""You are an expert AI research assistant.

Answer the question thoroughly and accurately using ONLY the context provided below.
If the answer cannot be determined from the context, state clearly:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:"""

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ DocAI Engine")
    st.caption("Private Apple Silicon RAG Stack")

    # System Status
    st.markdown("""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Inference Engine</div>
            <div class="sidebar-metric-value">Apple Metal (MLX)</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">LLM Model</div>
            <div class="sidebar-metric-value">Phi-3 Mini 4-bit</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Generation Speed</div>
            <div class="sidebar-metric-value">~56.8 tok/s</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Active Document Status
    if st.session_state.document:
        doc = st.session_state.document
        st.markdown("### 📄 Active Document")
        st.markdown(f"**File:** `{doc['name']}`")
        st.markdown(f"**Pages:** `{doc['total_pages']}` | **Chunks:** `{len(doc['all_chunks'])}`")

        if st.button("🗑️ Unload Document", use_container_width=True):
            st.session_state.document = None
            st.session_state.messages = []
            st.session_state.summary = None
            st.rerun()

    else:
        st.info("No document loaded yet. Upload a PDF or click below.")
        if st.button("📂 Load Sample Paper (Attention)", use_container_width=True):
            sample_path = "data/raw/sample.pdf"
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    index_document(f.read(), "Attention Is All You Need.pdf")
                st.rerun()

    if st.session_state.messages:
        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ==========================================
# MAIN INTERFACE
# ==========================================

# Top Branding
st.markdown("## ⚡ Intelligent Document AI")
st.markdown("""
<div class="badge-container">
    <span class="badge badge-green">● 100% Offline & Private</span>
    <span class="badge badge-purple">⚡ Apple M-Series MLX GPU</span>
    <span class="badge badge-slate">📄 FAISS Cosine Retrieval</span>
    <span class="badge badge-slate">🔒 Zero Cloud Telemetry</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------
# VIEW 1: UPLOAD SCREEN (If no doc loaded)
# ------------------------------------------
if not st.session_state.document:
    st.markdown("""
        <div class="doc-card">
            <div class="doc-title">Upload a Research Paper or Document</div>
            <div class="doc-meta">PDF format supported. All processing occurs locally in unified memory on your Mac.</div>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select PDF Document", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) > 0:
            with st.spinner("⚡ Extracting text, chunking, and computing embeddings on Apple Silicon GPU..."):
                doc_data = index_document(file_bytes, uploaded_file.name)
            if doc_data:
                st.success(f"Successfully indexed **{uploaded_file.name}** ({doc_data['total_pages']} pages, {len(doc_data['all_chunks'])} chunks).")
                st.rerun()
            else:
                st.error("Could not extract readable text from this PDF. Please verify it contains text and not only scanned images.")

# ------------------------------------------
# VIEW 2: ACTIVE DOCUMENT WORKSPACE
# ------------------------------------------
else:
    doc = st.session_state.document

    # Active Document Banner
    st.markdown(f"""
        <div class="doc-card">
            <div class="doc-title">📄 {doc['name']}</div>
            <div class="doc-meta">{doc['total_pages']} Pages &nbsp;·&nbsp; {len(doc['all_chunks'])} Indexed Chunks &nbsp;·&nbsp; Ready for instant querying</div>
        </div>
    """, unsafe_allow_html=True)

    # Quick Summary Expander
    with st.expander("📝 Document Executive Summary", expanded=False):
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
        else:
            if st.button("Generate Summary"):
                with st.spinner("Analyzing document sections..."):
                    summary_query = "Summarize the primary purpose, core methodology, key findings, and conclusions of this document."
                    q_emb = generate_embedding(summary_query)
                    distances, indices = search_similar_chunks(q_emb, doc["index"], top_k=7)
                    context = "\n\n".join([doc["all_chunks"][i]["text"] for i in indices[0]])
                    prompt = build_prompt(context, summary_query)

                summary_text = st.write_stream(stream_answer(prompt, max_tokens=600))
                st.session_state.summary = summary_text

    st.markdown("### 💬 Ask Questions")

    # Render Chat History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-card">{msg["content"]}</div>', unsafe_allow_html=True)
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Sources & Citations ({len(msg['sources'])} references)"):
                    for s in msg["sources"]:
                        st.markdown(f"""
                            <div class="citation-box">
                                <strong>Page {s['page']} (Chunk #{s['chunk_id']})</strong> — Match: {s['score']}%<br>
                                {s['text']}
                            </div>
                        """, unsafe_allow_html=True)

    # Input Form
    with st.form(key="query_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_query = st.text_input(
                "Ask anything about this document...",
                placeholder="e.g. What is the core problem addressed in this paper?",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Ask ⚡", use_container_width=True)

    # Query Execution
    if submitted and user_query.strip():
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query.strip()})
        st.markdown(f'<div class="user-bubble">{user_query.strip()}</div>', unsafe_allow_html=True)

        # Retrieve relevant chunks
        with st.spinner("Searching document context..."):
            query_embedding = generate_embedding(user_query.strip())
            distances, indices = search_similar_chunks(query_embedding, doc["index"], top_k=5)

            context_parts = []
            sources = []
            for rank, i in enumerate(indices[0]):
                chunk = doc["all_chunks"][i]
                score = round(float(distances[0][rank]) * 100, 1)
                context_parts.append(chunk["text"])
                sources.append({
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "score": score,
                    "text": chunk["text"]
                })

            context = "\n\n".join(context_parts)
            prompt = build_prompt(context, user_query.strip())

        # Assistant Card Container
        st.markdown('<div class="assistant-card">', unsafe_allow_html=True)
        full_answer = st.write_stream(stream_answer(prompt, max_tokens=500))
        st.markdown('</div>', unsafe_allow_html=True)

        # Render Sources for the answer
        if sources:
            with st.expander(f"📚 Sources & Citations ({len(sources)} references)"):
                for s in sources:
                    st.markdown(f"""
                        <div class="citation-box">
                            <strong>Page {s['page']} (Chunk #{s['chunk_id']})</strong> — Match: {s['score']}%<br>
                            {s['text']}
                        </div>
                    """, unsafe_allow_html=True)

        # Save assistant message with sources to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources
        })
