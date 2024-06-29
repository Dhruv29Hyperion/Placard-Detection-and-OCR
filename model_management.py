import cv2
import time
import torch
import os
import shutil
import pytesseract
from pytesseract import Output

# Set the path to the Tesseract executable (if not already in your PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path as necessary

# Function to clear directories
def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

# Directories to save the last detected image, cropped objects, and OCR results
answer_frame_dir = 'answer_frame'
text_detections_dir = 'text_detections'
ocr_results_dir = 'ocr_results'

# Clear pre-existing files in directories
clear_directory(answer_frame_dir)
clear_directory(text_detections_dir)
clear_directory(ocr_results_dir)

# Load the YOLOv5 models
print("Loading YOLOv5 models...")
placard_model = torch.hub.load('yolov5', 'custom', path='best_placard.pt', source='local')
text_model = torch.hub.load('yolov5', 'custom', path='best_text_v3.pt', source='local')
print("Models loaded successfully.")

# Function to detect placard in a frame
def detect_placard(frame):
    # Resize the frame to 640x640 as YOLOv5 expects this input size
    frame_resized = cv2.resize(frame, (640, 640))
    results = placard_model(frame_resized)  # Perform inference
    return results, frame_resized.shape[:2]

# Function to crop the detected object from the frame
def crop_detected_objects(original_frame, resized_shape, results):
    original_h, original_w = original_frame.shape[:2]
    resized_h, resized_w = resized_shape[:2]
    crops = []
    for *xyxy, conf, cls in results.xyxy[0]:
        # Scale the coordinates back to the original frame size
        x1 = int(xyxy[0] * (original_w / resized_w))
        y1 = int(xyxy[1] * (original_h / resized_h))
        x2 = int(xyxy[2] * (original_w / resized_w))
        y2 = int(xyxy[3] * (original_h / resized_h))
        crop = original_frame[y1:y2, x1:x2]
        crops.append((crop, (x1, y1, x2, y2)))
    return crops

# Function to detect text in a cropped image and save detected text regions
def detect_text(crop, crop_coords, idx):
    text_results = text_model(crop)
    crops = []
    text = False
    for *xyxy, conf, cls in text_results.xyxy[0]:
        # Scale the coordinates to the original cropped image size and expand the bounding box
        padding = 5  # Number of pixels to expand the bounding box
        x1 = max(0, int(xyxy[0]) - padding)
        y1 = max(0, int(xyxy[1]) - padding)
        x2 = min(crop.shape[1], int(xyxy[2]) + padding)
        y2 = min(crop.shape[0], int(xyxy[3]) + padding)
        text_crop = crop[y1:y2, x1:x2]
        crops.append(text_crop)
        text_crop_resized = cv2.resize(text_crop, (text_crop.shape[1] * 2, text_crop.shape[0] * 2), interpolation=cv2.INTER_LINEAR)  # Increase resolution
        text_crop_path = f'{text_detections_dir}/crop_{idx}_text_{len(crops)}.tiff'
        cv2.imwrite(text_crop_path, text_crop_resized)
        print(f"Cropped text saved: {text_crop_path}")

        # Perform OCR on the enhanced text image
        ocr_result = pytesseract.image_to_string(text_crop_resized, config='--psm 6', output_type=Output.STRING)
        if ocr_result.strip():  # Check if OCR result is not empty
            text = True
            ocr_path = f'{ocr_results_dir}/crop_{idx}_text_{len(crops)}.txt'
            with open(ocr_path, 'w') as f:
                f.write(ocr_result)
            print(f"OCR result saved: {ocr_path}")
        else:
            print("OCR result is empty, not saved.")

    # If no text was detected, run OCR on the entire placard
    if not crops or not text:
        print("No text detected by the model. Running OCR on the entire placard.")
        placard_crop_resized = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        ocr_result = pytesseract.image_to_string(placard_crop_resized, config='--psm 6', output_type=Output.STRING)
        if ocr_result.strip():
            ocr_path = f'{ocr_results_dir}/crop_{idx}_full_placard.txt'
            with open(ocr_path, 'w') as f:
                f.write(ocr_result)
            print(f"OCR result for full placard saved: {ocr_path}")
        else:
            print("OCR result for full placard is empty, not saved.")

    return crops

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam initialized. Starting video capture...")
print("Starting detection...")

# Set duration for the webcam to run (in seconds)
total_duration = 30
start_time = time.time()

# Variables to check continuous detection
detection_start_time = None
detection_threshold = 5  # 5 seconds for continuous detection
last_detected_image = None

while time.time() - start_time < total_duration:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform detection
    results, resized_shape = detect_placard(frame)

    # Check if a placard is detected
    if len(results.xyxy[0]) > 0:  # If placard is detected
        if detection_start_time is None:
            detection_start_time = time.time()
            print("Placard detected. Starting detection timer...")
        elif time.time() - detection_start_time >= detection_threshold:
            print(f"Placard detected continuously for {detection_threshold} seconds.")
            last_detected_image = frame

            # Crop and save the detected objects
            crops = crop_detected_objects(frame, resized_shape, results)
            for idx, (crop, crop_coords) in enumerate(crops):
                crop_path = f'{answer_frame_dir}/crop_{idx}.png'
                cv2.imwrite(crop_path, crop)
                print(f"Cropped object saved: {crop_path}")

                # Perform text detection on the cropped image and save text crops
                detect_text(crop, crop_coords, idx)
            break
    else:
        detection_start_time = None
        print("No placard detected. Resetting detection timer.")

    # Display the frame with detection results
    results.render()  # Render the bounding boxes on the frame
    frame_with_results = results.ims[0]
    cv2.imshow('Webcam', frame_with_results)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting video capture.")
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Save the last detected image if it exists
if last_detected_image is not None:
    cv2.imwrite(f'{answer_frame_dir}/answer.png', last_detected_image)
    print("Last detected image saved.")
else:
    print("No placard detected for 5 seconds continuously.")


