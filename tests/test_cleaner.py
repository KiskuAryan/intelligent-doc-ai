import pytest
from utils.cleaner import clean_text

def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""

def test_clean_text_removes_special_tokens():
    raw = "Hello world <EOS> this is a test <pad> with tokens <unk> <bos>."
    cleaned = clean_text(raw)
    assert "<EOS>" not in cleaned
    assert "<pad>" not in cleaned
    assert "<unk>" not in cleaned
    assert "<bos>" not in cleaned
    assert "Hello world this is a test with tokens ." == cleaned

def test_clean_text_removes_chat_markers():
    raw = "<|user|> Tell me something <|end|> <|assistant|> Here is the answer."
    cleaned = clean_text(raw)
    assert "<|user|>" not in cleaned
    assert "<|end|>" not in cleaned
    assert "<|assistant|>" not in cleaned
    assert "Tell me something Here is the answer." == cleaned

def test_clean_text_removes_isolated_page_numbers():
    raw = """Title of Document
42
This is section 1.
    105   
This is section 2."""
    cleaned = clean_text(raw)
    assert "42" not in cleaned.split()
    assert "105" not in cleaned.split()
    assert "Title of Document This is section 1. This is section 2." == cleaned

def test_clean_text_deduplicates_consecutive_lines():
    raw = """Header Text
Header Text
Body paragraph 1.
Body paragraph 2.
Footer Text
Footer Text"""
    cleaned = clean_text(raw)
    assert cleaned == "Header Text Body paragraph 1. Body paragraph 2. Footer Text"

def test_clean_text_normalizes_whitespace():
    raw = "Multiple    spaces \t and \n\n newlines  here.   "
    cleaned = clean_text(raw)
    assert cleaned == "Multiple spaces and newlines here."
