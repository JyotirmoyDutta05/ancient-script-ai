import io
import cv2
import json
from google.cloud import vision
from PIL import Image
import numpy as np
from pathlib import Path

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image_path):
    """Enhance readability: grayscale, denoise, contrast."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    # Adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    processed_path = Path("temp_preprocessed.png")
    cv2.imwrite(str(processed_path), gray)
    return processed_path


# -------------------------------
# OCR with Google Vision API
# -------------------------------
def extract_text_with_bboxes(image_path):
    """Extract text + bounding boxes from Google Vision OCR."""
    client = vision.ImageAnnotatorClient()

    # Preprocess image
    processed_path = preprocess_image(image_path)

    with io.open(processed_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"Google Vision error: {response.error.message}")

    annotations = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_text = ""
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    block_text += word_text + " "
            # Extract bounding box coordinates
            vertices = [(v.x, v.y) for v in block.bounding_box.vertices]
            annotations.append({
                "text": block_text.strip(),
                "bbox": vertices
            })

    extracted_text = response.full_text_annotation.text.strip() if response.full_text_annotation.text else ""

    return {
        "full_text": extracted_text,
        "blocks": annotations
    }


# -------------------------------
# Optional visualization
# -------------------------------
def visualize_bboxes(image_path, blocks, output_path="ocr_boxes_preview.jpg"):
    img = cv2.imread(str(image_path))
    for b in blocks:
        pts = np.array(b["bbox"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    cv2.imwrite(output_path, img)
    print(f"🟩 Bounding boxes saved as {output_path}")


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    image_path = "/Users/jyotirmoy/Desktop/Image/ancient-script-ai/data/samples/3.png"

    print("[INFO] Performing OCR with bounding boxes ...")
    result = extract_text_with_bboxes(image_path)

    print("\n🕉️ Extracted Text:\n")
    print(result["full_text"])

    # Save structured JSON
    output_json = Path("ocr_output.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"\n📄 Structured OCR saved as: {output_json.resolve()}")

    # Optional: visualize boxes
    visualize_bboxes(image_path, result["blocks"])
