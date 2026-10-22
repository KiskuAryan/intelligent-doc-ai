import pytest
from utils.chunker import chunk_text

def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("    ") == []
    assert chunk_text(None) == []

def test_chunk_text_smaller_than_chunk_size():
    text = "Short text with only six words."
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_text_sliding_window_overlap():
    words = [f"word{i}" for i in range(10)]
    text = " ".join(words)
    # chunk_size = 5, overlap = 2 -> step = 3
    # chunk 0: words 0..4 (5 words)
    # chunk 1: words 3..7 (5 words)
    # chunk 2: words 6..9 (4 words)
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert len(chunks) == 3
    assert chunks[0] == "word0 word1 word2 word3 word4"
    assert chunks[1] == "word3 word4 word5 word6 word7"
    assert chunks[2] == "word6 word7 word8 word9"

def test_chunk_text_invalid_parameters():
    text = "Some sample text"
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text(text, chunk_size=0, overlap=0)

    with pytest.raises(ValueError, match="overlap cannot be negative"):
        chunk_text(text, chunk_size=10, overlap=-1)

    with pytest.raises(ValueError, match="strictly greater than overlap"):
        chunk_text(text, chunk_size=10, overlap=10)

    with pytest.raises(ValueError, match="strictly greater than overlap"):
        chunk_text(text, chunk_size=10, overlap=15)
