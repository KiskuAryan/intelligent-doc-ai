from sentence_transformers import SentenceTransformer

# Lazy-loaded singleton — model won't load until first call
_model = None

def get_model():
    """
    Returns the embedding model, loading it only once (lazy init).
    Runs on Apple Silicon GPU (mps) automatically.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _model

def generate_embedding(text):
    """
    Generates a normalized L2 embedding for a single text string.

    Args:
        text : input string

    Returns:
        numpy array of shape (384,)
    """
    model = get_model()
    return model.encode(text, normalize_embeddings=True)

def generate_embeddings(texts, batch_size=32):
    """
    Generates normalized L2 embeddings for a list of texts in parallel batches.
    Significantly faster for document indexing than one-by-one encoding.

    Args:
        texts : list of strings
        batch_size : batch size for GPU parallel encoding

    Returns:
        numpy array of shape (N, 384)
    """
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
