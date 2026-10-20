from utils.pdf_parser import extract_text_from_pdf
from utils.cleaner import clean_text
from utils.chunker import chunk_text
from utils.embedder import generate_embedding, generate_embeddings
from utils.retriever import create_faiss_index, search_similar_chunks
from utils.generator import stream_answer, generate_answer

# =========================
# CONFIG
# =========================

PDF_PATH = "data/raw/sample.pdf"
QUERY    = "What is self-attention?"
TOP_K    = 5

# =========================
# STEP 1 — PARSE PDF
# =========================

print("\n[1/5] Extracting text from PDF...")
pages = extract_text_from_pdf(PDF_PATH)
print(f"      {len(pages)} pages extracted.")

# =========================
# STEP 2 — CLEAN + CHUNK + BATCH EMBED
# =========================

print("[2/5] Cleaning, chunking, and batch embedding on Apple Silicon GPU...")

all_chunks = []
chunk_texts = []

for page in pages:
    cleaned = clean_text(page["text"])
    chunks  = chunk_text(cleaned, chunk_size=300, overlap=50)

    for idx, chunk in enumerate(chunks):
        if len(chunk.strip()) < 20:
            continue

        chunk_texts.append(chunk)
        all_chunks.append({
            "page":     page["page"],
            "chunk_id": idx,
            "text":     chunk
        })

if len(chunk_texts) == 0:
    print("[ERROR] No valid text found in PDF.")
    exit()

embeddings = generate_embeddings(chunk_texts, batch_size=32)
print(f"      {len(all_chunks)} valid chunks | {len(embeddings)} embeddings generated.")

# =========================
# STEP 3 — BUILD FAISS INDEX
# =========================

print("[3/5] Building FAISS index (cosine similarity)...")
index = create_faiss_index(embeddings)
print("      Index ready.")

# =========================
# STEP 4 — RETRIEVE
# =========================

print(f"\n[4/5] Query: \"{QUERY}\"")
query_embedding         = generate_embedding(QUERY)
distances, indices      = search_similar_chunks(query_embedding, index, top_k=TOP_K)

print(f"\n      Top {TOP_K} matching chunks:\n")

context = ""
for rank, i in enumerate(indices[0]):
    score = round(float(distances[0][rank]) * 100, 1)
    chunk = all_chunks[i]
    print(f"  Rank {rank+1} | Page {chunk['page']} | Chunk {chunk['chunk_id']} | Score: {score}%")
    print(f"  {chunk['text'][:200]}...")
    print()
    context += chunk["text"] + "\n\n"

# =========================
# STEP 5 — GENERATE ANSWER (STREAMING)
# =========================

print("[5/5] Generating answer (MLX Metal 50+ tok/s)...\n")

prompt = f"""You are an expert AI research assistant.

Answer the question using ONLY the context provided below.
If the answer is not found in the context, say exactly:
"I could not find the answer in the document."

Be concise. Explain technical terms clearly.

Context:
{context}

Question:
{QUERY}

Answer:"""

print("=" * 60)
print("GENERATED ANSWER")
print("=" * 60)

for token in stream_answer(prompt, max_tokens=400):
    print(token, end="", flush=True)

print("\n" + "=" * 60)
