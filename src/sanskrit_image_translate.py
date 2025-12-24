import io
import os
import re
import unicodedata
from typing import List
from google.cloud import vision
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# PATHS
# -----------------------------
PROJECT_ROOT = "/Users/jyotirmoy/Desktop/Image/ancient-script-ai"
MODEL_DIR = f"{PROJECT_ROOT}/models/nllb_frozen"
IMAGE_PATH = f"{PROJECT_ROOT}/data/samples/2.png"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/jyotirmoy/Desktop/vision-key.json"

# -----------------------------
# OCR
# -----------------------------
def ocr_sanskrit_google(image_path: str) -> str:
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    resp = client.text_detection(image=image)
    if resp.error.message:
        raise RuntimeError(resp.error.message)
    return resp.text_annotations[0].description

# -----------------------------
# TEXT NORMALIZATION
# -----------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s{2,}", " ", text)
    # Normalize danda and spacing
    text = re.sub(r"[|]{2,}", "॥", text)
    text = re.sub(r"\s*॥\s*", " ॥ ", text)
    text = re.sub(r"\s*।\s*", " । ", text)
    # Remove stray non-Devanagari artifacts while keeping danda
    text = re.sub(r"[^\u0900-\u097F\s।॥\-–—]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def chunk_by_punctuation(text: str) -> List[str]:
    parts = re.split(r"(?:\s*॥\s*|\s*।\s*)", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return [c for c in chunks if len(c) >= 6]

# -----------------------------
# MODEL LOAD
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.eval()

# Source Sanskrit
tokenizer.src_lang = "san_Deva"
# Resolve language ids
def lang_id(code: str):
    try:
        tid = tokenizer.convert_tokens_to_ids(code)
        if tid is None or tid == tokenizer.unk_token_id:
            tid = getattr(tokenizer, "lang_code_to_id", {}).get(code)
        return tid
    except Exception:
        return getattr(tokenizer, "lang_code_to_id", {}).get(code)

ENG_ID = lang_id("eng_Latn")
HIN_ID = lang_id("hin_Deva")
if ENG_ID is None or HIN_ID is None:
    raise RuntimeError("Could not resolve language ids for eng_Latn or hin_Deva.")

# -----------------------------
# GENERATION HELPERS
# -----------------------------
def generate(text: str, bos_id: int, mode: str = "beam", max_len: int = 256) -> str:
    # Set forced BOS dynamically per target language
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_bos_token_id = bos_id
    else:
        model.config.forced_bos_token_id = bos_id

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        if mode == "beam":
            out = model.generate(
                **inputs,
                max_length=max_len,
                num_beams=6,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
                early_stopping=True,
                min_length=25
            )
        else:
            out = model.generate(
                **inputs,
                max_length=max_len,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=3,
                encoder_no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                early_stopping=True,
                min_length=25
            )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def suspicious(text: str) -> bool:
    if len(text.strip()) < 40:
        return True
    if re.search(r"(\b\w+\b)(?:\s+\1){2,}", text):
        return True
    if len(re.findall(r",", text)) >= 8:
        return True
    return False

def translate_chunk_safely(ch: str) -> str:
    # Direct Sanskrit → English
    direct = generate(ch, bos_id=ENG_ID, mode="beam")
    if suspicious(direct):
        alt = generate(ch, bos_id=ENG_ID, mode="sample")
        if not suspicious(alt):
            direct = alt

    # Pivot Sanskrit → Hindi → English
    hin = generate(ch, bos_id=HIN_ID, mode="beam")
    if suspicious(hin):
        hin = generate(ch, bos_id=HIN_ID, mode="sample")

    pivot = generate(hin, bos_id=ENG_ID, mode="beam")
    if suspicious(pivot):
        alt_pivot = generate(hin, bos_id=ENG_ID, mode="sample")
        if not suspicious(alt_pivot):
            pivot = alt_pivot

    # Choose cleaner output
    candidates = [direct, pivot]
    # Heuristic: prefer longer, less repetitive, fewer comma chains
    def score(t):
        length = len(t)
        reps = 1 + len(re.findall(r"(\b\w+\b)(?:\s+\1){2,}", t))
        commas = 1 + len(re.findall(r",", t))
        return length / (reps * commas)
    best = max(candidates, key=score)

    # Trim pathological enumerations as last resort
    if suspicious(best):
        best = re.sub(r"(,[^,]{0,20}){8,}", "", best)
    return best.strip()

# -----------------------------
# PIPELINE
# -----------------------------
def main():
    print("Running Google Vision OCR...")
    raw = ocr_sanskrit_google(IMAGE_PATH)
    text = normalize_text(raw)

    print("\nOCR Sanskrit:")
    print(text)

    chunks = chunk_by_punctuation(text) or [text]
    translations = [translate_chunk_safely(ch) for ch in chunks]
    final = " ".join(t for t in translations if t)

    print("\nEnglish Translation:")
    print(final)

if __name__ == "__main__":
    main()
