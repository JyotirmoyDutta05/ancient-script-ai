import unicodedata
import re

def normalize_ocr_text(text: str) -> str:
    """
    Minimal normalization:
    - Unicode NFC
    - Remove excessive spaces
    - Preserve danda and verse markers
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Remove extra spaces but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Clean spacing around danda
    text = re.sub(r"\s*॥\s*", " ॥ ", text)
    text = re.sub(r"\s*।\s*", " । ", text)

    return text.strip()
