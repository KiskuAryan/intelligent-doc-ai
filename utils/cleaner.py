import re

def clean_text(text):
    """
    Cleans raw PDF-extracted text:
    - Removes special model tokens (<EOS>, <pad>)
    - Strips invalid unicode bytes
    - Removes isolated standalone page numbers
    - Deduplicates consecutive duplicate lines (headers/footers)
    - Normalizes whitespace without corrupting punctuation
    """
    if not text:
        return ""

    # Remove special model tokens and chat template markers
    text = re.sub(r'<\|[^|]+?\|>|<EOS>|<pad>|<unk>|<bos>', ' ', text, flags=re.IGNORECASE)

    # Strip invalid unicode bytes
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    # Remove isolated page numbers (lines that only contain digits)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Split into lines and filter consecutive duplicates
    raw_lines = text.splitlines()
    clean_lines = []
    prev_line = None

    for line in raw_lines:
        trimmed = re.sub(r'[ \t]+', ' ', line).strip()
        if trimmed and trimmed != prev_line:
            clean_lines.append(trimmed)
            prev_line = trimmed

    # Join and collapse excessive spaces
    cleaned = " ".join(clean_lines)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)

    return cleaned.strip()
