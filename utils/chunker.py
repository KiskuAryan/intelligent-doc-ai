def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping word-based chunks.

    Args:
        text       : cleaned input string
        chunk_size : number of words per chunk (default 300)
        overlap    : number of words shared between consecutive chunks (default 50)

    Returns:
        List of chunk strings
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be strictly greater than overlap")

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # If we've reached the end, stop
        if end == len(words):
            break

        # Slide forward by (chunk_size - overlap) to create overlap
        start += chunk_size - overlap

    return chunks