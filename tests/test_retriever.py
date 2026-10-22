import pytest
import numpy as np
import faiss
from utils.retriever import create_faiss_index, search_similar_chunks

def test_create_faiss_index_basic():
    # 3 embeddings of dim 4
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    embeddings = [v1, v2, v3]

    index = create_faiss_index(embeddings)
    assert index.ntotal == 3

def test_create_faiss_index_empty():
    with pytest.raises(ValueError, match="Cannot create FAISS index from empty"):
        create_faiss_index([])

def test_search_similar_chunks():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([0.8, 0.2, 0.0], dtype=np.float32)
    embeddings = [v1, v2, v3]

    index = create_faiss_index(embeddings)

    # Query closest to v1
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    distances, indices = search_similar_chunks(query, index, top_k=2)

    assert len(indices[0]) == 2
    # The closest should be v1 (index 0) or v3 (index 2)
    assert indices[0][0] in [0, 2]
    # Cosine score should be high (close to 1.0)
    assert distances[0][0] > 0.9

def test_search_similar_chunks_fewer_than_top_k():
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    index = create_faiss_index([v1])

    query = np.array([1.0, 0.0], dtype=np.float32)
    # top_k is 5, but total is 1
    distances, indices = search_similar_chunks(query, index, top_k=5)

    assert len(indices[0]) == 1
    assert indices[0][0] == 0
    assert distances[0][0] == pytest.approx(1.0, abs=1e-4)
