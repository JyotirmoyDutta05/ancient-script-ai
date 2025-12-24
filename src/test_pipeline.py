from src.pipeline.image_to_translation import ImageTranslationPipeline

PROJECT_ID = "ancient-script-ai"
IMAGE_PATH = "data/samples/3.png"

pipeline = ImageTranslationPipeline(PROJECT_ID)
result = pipeline.process_image(IMAGE_PATH)

print("\n--- Extracted Text ---\n")
print(result["extracted_text"])

print("\n--- Translation ---\n")
print(result["translated_text"])
