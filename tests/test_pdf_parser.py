import pytest
import pymupdf as fitz
from utils.pdf_parser import extract_text_from_pdf

def test_extract_text_from_pdf_bytes():
    # Create in-memory single-page PDF with fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "DocAI automated test page 1")
    pdf_bytes = doc.tobytes()
    doc.close()

    extracted = extract_text_from_pdf(pdf_bytes)
    assert len(extracted) == 1
    assert extracted[0]["page"] == 1
    assert "DocAI automated test page 1" in extracted[0]["text"]

def test_extract_text_from_multi_page_pdf():
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "First page content")
    p2 = doc.new_page()
    p2.insert_text((50, 72), "Second page content")
    pdf_bytes = doc.tobytes()
    doc.close()

    extracted = extract_text_from_pdf(pdf_bytes)
    assert len(extracted) == 2
    assert extracted[0]["page"] == 1
    assert "First page" in extracted[0]["text"]
    assert extracted[1]["page"] == 2
    assert "Second page" in extracted[1]["text"]
