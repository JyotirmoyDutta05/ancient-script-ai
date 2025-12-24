from src.utils.text_normalization import normalize_ocr_text

CORPUS_PHRASES = {
    "श्रीगणेशाय नमः": "Salutations to Lord Ganesha",
    "अथरात्रिसूक्तम्": "The Hymn of Night",
}

def apply_phrase_corrections(original, translated):
    norm_original = normalize_ocr_text(original)
    for src, tgt in CORPUS_PHRASES.items():
        if normalize_ocr_text(src) in norm_original:
            # Replace only the matching phrase, not the whole translation
            return translated.replace(normalize_ocr_text(src), tgt)
    return translated
