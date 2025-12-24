from google.cloud import vision
import io
import cv2
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """Enhance readability: grayscale, contrast, denoise."""
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=30)
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    cv2.imwrite("temp_preprocessed.png", gray)
    return "temp_preprocessed.png"

def extract_text_google(image_path):
    """Extract text using Google Cloud Vision API."""
    client = vision.ImageAnnotatorClient()
    
    # Preprocess image
    processed_path = preprocess_image(image_path)

    with io.open(processed_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Run OCR
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"Google Vision error: {response.error.message}")

    # The first element is the full text
    extracted_text = texts[0].description if texts else ""
    return extracted_text.strip()

if __name__ == "__main__":
    path = "/Users/jyotirmoy/Desktop/Image/ancient-script-ai/data/samples/4.png"
    text = extract_text_google(path)
    print("🕉️ Extracted Text:\n")
    print(text)
