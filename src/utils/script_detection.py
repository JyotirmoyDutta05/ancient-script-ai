def detect_script(text: str) -> str:
    """
    Detect script based on Unicode ranges.
    Returns: 'sa' (Sanskrit), 'ta' (Tamil), or None
    """
    for char in text:
        code = ord(char)
        # Devanagari
        if 0x0900 <= code <= 0x097F:
            return "sa"
        # Tamil
        if 0x0B80 <= code <= 0x0BFF:
            return "ta"
    return None
