import pytesseract
from pytesseract import Output
from PIL import Image

# Set the path to the Tesseract executable (if not already in your PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as necessary

# Load the image from the file
image_path = '1.png'
image = Image.open(image_path)

# Perform OCR on the image with Malayalam language specified
ocr_result = pytesseract.image_to_string(image, lang='mal', config='--psm 6', output_type=Output.STRING)

# Write the OCR result to a text file
with open('ocr_result.txt', 'w', encoding='utf-8') as f:
    f.write(ocr_result)
