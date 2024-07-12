from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import cv2
import torch
import os
import pytesseract
from pytesseract import Output
from collections import Counter
import numpy as np
from fuzzywuzzy import process

app = FastAPI()

# Set the path to the Tesseract executable (if not already in your PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as necessary

# Load the YOLOv5 models
placard_model = torch.hub.load('yolov5', 'custom', path='best_placard.pt', source='local')
text_model = torch.hub.load('yolov5', 'custom', path='best_text_v3.pt', source='local')


# Function to detect placard in a frame
def detect_placard(frame):
    frame_resized = cv2.resize(frame, (640, 640))
    results = placard_model(frame_resized)
    return results, frame_resized.shape[:2]


# Function to crop detected objects from the frame
def crop_detected_objects(original_frame, resized_shape, results):
    original_h, original_w = original_frame.shape[:2]
    resized_h, resized_w = resized_shape[:2]
    crops = []
    for *xyxy, conf, cls in results.xyxy[0]:
        x1 = int(xyxy[0] * (original_w / resized_w))
        y1 = int(xyxy[1] * (original_h / resized_h))
        x2 = int(xyxy[2] * (original_w / resized_w))
        y2 = int(xyxy[3] * (original_h / resized_h))
        crop = original_frame[y1:y2, x1:x2]
        crops.append(crop)
    return crops


# Function to detect text in a cropped image and save detected text regions
def detect_text(crop, language):
    text_results = text_model(crop)
    texts = []
    for *xyxy, conf, cls in text_results.xyxy[0]:
        padding = 5
        x1 = max(0, int(xyxy[0]) - padding)
        y1 = max(0, int(xyxy[1]) - padding)
        x2 = min(crop.shape[1], int(xyxy[2]) + padding)
        y2 = min(crop.shape[0], int(xyxy[3]) + padding)
        text_crop = crop[y1:y2, x1:x2]
        text_crop_resized = cv2.resize(text_crop, (text_crop.shape[1] * 2, text_crop.shape[0] * 2),
                                       interpolation=cv2.INTER_LINEAR)

        if language == "mal":
            ocr_result = pytesseract.image_to_string(text_crop_resized, config='--psm 6', lang='mal',
                                                     output_type=Output.STRING)
        else:
            ocr_result = pytesseract.image_to_string(text_crop_resized, config='--psm 6', output_type=Output.STRING)

        if ocr_result.strip():
            texts.append(ocr_result.strip())

    if not texts:
        placard_crop_resized = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        if language == "mal":
            ocr_result = pytesseract.image_to_string(placard_crop_resized, config='--psm 5', lang='mal',
                                                     output_type=Output.STRING)
        else:
            ocr_result = pytesseract.image_to_string(placard_crop_resized, config='--psm 5', output_type=Output.STRING)

        if ocr_result.strip():
            texts.append(ocr_result.strip())

    return texts


def find_closest_match(text, class_name):
    file_path = f'text_directory/{class_name}.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            options = file.read().splitlines()
        closest_match, _ = process.extractOne(text, options)
        return closest_match


@app.post("/detect_placard/")
async def detect_placard_api(images: List[UploadFile] = File(...), timer: int = Form(...), language: str = Form(...),
                             class_name: str = Form(...)):
    frames = []
    for image in images:
        file_bytes = await image.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames.append(frame)

    detected_texts = []
    placard_detected = False

    for frame in frames:
        results, resized_shape = detect_placard(frame)
        if results.xyxy[0].shape[0] > 0:  # Check if any placard is detected
            placard_detected = True
            if timer == 3:
                crops = crop_detected_objects(frame, resized_shape, results)
                for crop in crops:
                    texts = detect_text(crop, language)
                    detected_texts.extend(texts)
            break  # Exit the loop once a placard is detected

    response = {
        "placard_detected": placard_detected,
        "text": ""
    }

    if placard_detected and timer == 3:
        if detected_texts:
            most_common_text = Counter(detected_texts).most_common(1)[0][0]
            # Don't send newline characters or '-' in response
            most_common_text = most_common_text.replace('\n', '').replace('-', '')
            most_common_text = find_closest_match(most_common_text, class_name)
            response["text"] = most_common_text

    return JSONResponse(content=response)
