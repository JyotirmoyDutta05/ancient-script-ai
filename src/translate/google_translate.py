from google.cloud import translate_v3 as translate

class GoogleTranslator:
    def __init__(self, project_id, location="global"):
        self.client = translate.TranslationServiceClient()
        self.parent = f"projects/{project_id}/locations/{location}"

    def translate_lines(
        self,
        lines,
        source_lang,
        target_lang="en"
    ):
        translated_lines = []

        for line in lines:
            if not line.strip():
                translated_lines.append("")
                continue

            response = self.client.translate_text(
                parent=self.parent,
                contents=[line],
                mime_type="text/plain",
                source_language_code=source_lang,
                target_language_code=target_lang,
            )

            translated_lines.append(
                response.translations[0].translated_text
            )

        return translated_lines
