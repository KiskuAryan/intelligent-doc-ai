import pymupdf as fitz  # PyMuPDF

def extract_text_from_pdf(pdf_input):
    """
    Extracts text from each page of a PDF using PyMuPDF.

    Args:
        pdf_input : file path (str) OR raw bytes of the PDF

    Returns:
        List of dicts: [{"page": int, "text": str}, ...]
    """
    if isinstance(pdf_input, (bytes, bytearray)):
        doc = fitz.open(stream=pdf_input, filetype="pdf")
    else:
        doc = fitz.open(pdf_input)

    pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        pages.append({
            "page": page_num + 1,
            "text": text
        })

    doc.close()

    return pages
