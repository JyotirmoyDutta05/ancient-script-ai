print(">>> USING UPDATED IMAGE TRANSLATION PIPELINE <<<")

from src.ocr.ocr_google_vision_bboxes import extract_text_with_bboxes
from src.translate.google_translate import GoogleTranslator
from src.utils.text_normalization import normalize_ocr_text
from src.utils.script_detection import detect_script
from src.postprocess.corpus_phrases import apply_phrase_corrections

class ImageTranslationPipeline:
    def __init__(self, project_id):
        self.translator = GoogleTranslator(project_id)

    def process_image(self, image_path):
        # OCR
        ocr_result = extract_text_with_bboxes(image_path)
        raw_text = ocr_result["full_text"]

        # Normalize
        clean_text = normalize_ocr_text(raw_text)

        # Split into lines
        lines = clean_text.split("\n")

        # Detect language
        source_lang = detect_script(clean_text)
        if source_lang is None:
            raise ValueError("Unable to detect script/language")

        # Translate line-by-line
        translated_lines = self.translator.translate_lines(
            lines=lines,
            source_lang=source_lang
        )

        # Post-process with corpus phrases
        final_translation = []
        for src, tgt in zip(lines, translated_lines):
            corrected = apply_phrase_corrections(src, tgt)
            final_translation.append(corrected)

        return {
            "image": str(image_path),
            "source_language": source_lang,
            "extracted_text": clean_text,
            "translated_text": "\n".join(final_translation),
            "blocks": ocr_result["blocks"],
        }
