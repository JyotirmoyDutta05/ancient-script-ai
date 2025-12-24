import json
from pathlib import Path
import collections

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "models"

def load_language_model(lang_code: str):
    word_file = MODELS_ROOT / f"{lang_code}_word_freq.json"
    char_file = MODELS_ROOT / f"{lang_code}_char_trigrams.json"

    if not word_file.exists() or not char_file.exists():
        raise FileNotFoundError(f"Models for {lang_code} not found in {MODELS_ROOT}")

    with open(word_file, "r", encoding="utf-8") as f:
        word_freq = collections.Counter(json.load(f))
    with open(char_file, "r", encoding="utf-8") as f:
        char_grams = collections.Counter(json.load(f))

    return word_freq, char_grams

# Usage:
# sa_words, sa_chars = load_language_model("sa")
# hi_words, hi_chars = load_language_model("hi")
