import faiss
import numpy as np

def create_faiss_index(embeddings):
    """
    Builds a FAISS index using Inner Product (cosine similarity).

    Since BGE embeddings are L2-normalized (done in embedder.py),
    IndexFlatIP is equivalent to cosine similarity — which is what
    BGE models are trained and optimized for.

    Args:
        embeddings : list of numpy arrays (each shape: (384,))

    Returns:
        faiss.IndexFlatIP index
    """
    dimension = embeddings[0].shape[0]

    # Inner Product index — correct for normalized BGE embeddings
    index = faiss.IndexFlatIP(dimension)

    embedding_matrix = np.array(embeddings).astype("float32")

    # Normalize just in case (safe to call even if already normalized)
    faiss.normalize_L2(embedding_matrix)

    index.add(embedding_matrix)

    return index


def search_similar_chunks(query_embedding, index, top_k=5):
    """
    Retrieves top-k most similar chunks using cosine similarity.

    Args:
        query_embedding : numpy array (shape: (384,))
        index           : faiss.IndexFlatIP
        top_k           : number of results to return

    Returns:
        distances : cosine similarity scores (higher = more similar)
        indices   : positions in the original chunks list
    """
    query_vector = np.array([query_embedding]).astype("float32")

    # Normalize query for cosine similarity
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    return distances, indices