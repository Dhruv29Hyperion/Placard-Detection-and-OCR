# Placard Detection and OCR System

## Overview

This system consists of two main components:
1. **FastAPI Application**: Detects placards and performs OCR on the detected placards.
2. **Camera Control Script**: Captures frames from a webcam and sends them to the FastAPI application for processing every second.

## Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Libraries:
  - `fastapi`
  - `uvicorn`
  - `opencv-python`
  - `torch`
  - `pytesseract`
  - `numpy`
  - `fuzzywuzzy`
  - `python-Levenshtein`
  - `requests`
- Malayalam Data for Tesseract:
  - Download the Malayalam language data file from the [tesseract-ocr/tessdata repository](https://github.com/tesseract-ocr/tessdata) and place it in your Tesseract installation folder.

Install the required libraries using:
```bash
pip install fastapi uvicorn opencv-python torch pytesseract numpy fuzzywuzzy python-Levenshtein requests
```

## Picture Dataset

Download the dataset from the following link and place it in your working directory:
[Picture Dataset](https://drive.google.com/drive/folders/1XrR2sq9PVeCRCgOZLaWBKdbs9Sjvvg6m?usp=sharing)

## Project Components

### FastAPI Application

- **Purpose**: Detects placards in images and performs OCR on detected placards.
- **Files**:
  - `placard_detection_api.py`: The main FastAPI application script.
  - `text_directory/`: Directory containing text files for different classes.
- **Models**: YOLOv5 models for placard detection (`best_placard.pt`) and text detection (`best_text_v3.pt`).

**API Endpoint**: `/detect_placard/`
  - **Inputs**:
    - `images`: List of image files.
    - `timer`: Timer value to determine when to perform OCR.
    - `language`: Language for OCR (e.g., "eng", "mal").
    - `class_name`: Class name for fuzzy text matching.
  - **Outputs**: JSON response indicating placard detection status and recognized text.

### Camera Control Script

- **Purpose**: Captures frames from a webcam and sends them to the FastAPI application for processing.
- **Files**:
  - `camera_control.py`: Script to capture frames and call the API.

### Training the Models

Use the provided Jupyter notebook to train the models:
- `Placard_&_Text_Detection_Training.ipynb`

### Data Files

- `custom_data.yaml`: Configuration file for the dataset.
- `best_placard.pt`: Trained YOLOv5 model for placard detection.
- `best_text_v3.pt`: Trained YOLOv5 model for text detection.

## Setup and Execution

### Set Up the FastAPI Application

1. **Directory Structure**:
   - Create `fastapi_app` directory.
   - Inside `fastapi_app`, create `text_directory` subdirectory.
   - Place `placard_detection_api.py` in `fastapi_app`.
   - Add text files to `text_directory` for different classes, each containing possible text options, one per line.

2. **Run the FastAPI Application**:
   ```bash
   cd fastapi_app
   uvicorn placard_detection_api:app --reload
   ```

### Set Up and Run the Camera Control Script

1. **Directory Structure**:
   - Create `camera_control` directory.
   - Place `camera_control.py` in `camera_control`.

2. **Run the Camera Control Script**:
   ```bash
   cd camera_control
   python camera_control.py
   ```

## Detailed Workflow

### FastAPI Application

1. **Receives images and parameters** from the camera control script.
2. **For each image**:
   - **Detect Placard**: The `detect_placard` function resizes the frame and uses the YOLOv5 model to detect placards.
   - **Crop Detected Placards**: The `crop_detected_objects` function crops the detected placards from the original frame based on detection results.
   - **Perform OCR**: If `timer` equals 3, the `detect_text` function performs OCR on the cropped placard and extracts text.
     - If the language is set to Malayalam (`mal`), the OCR process uses the Malayalam language data.
   - **Fuzzy Matching**: The `find_closest_match` function uses fuzzy matching to find the closest match for the detected text from a predefined list in a text file.
3. **Returns JSON response** indicating whether a placard was detected and the recognized text if applicable.

### Camera Control Script

1. **Initializes webcam**: The script initializes the webcam for capturing video frames.
2. **Captures frames** at a rate of 10 frames per second.
3. **Saves frames** as TIFF files in a temporary directory.
4. **Every second**:
   - **Collects the last 10 frames**.
   - **Prepares files for API request**: Reads the frames and sends them to the FastAPI application.
   - **Sends frames to API**: Makes a POST request to the FastAPI endpoint with the frames and additional parameters (timer, language, class_name).
   - **Handles API response**: Processes the response from the FastAPI application, printing whether a placard was detected and the recognized text if `timer` equals 3.
   - **Cleans up sent frames**: Deletes the frames after sending them to maintain directory cleanliness.
5. **Releases webcam** and performs final cleanup after the total duration elapses.

## Example Data Files

- `custom_data.yaml`: Configuration file for the YOLOv5 dataset.
  ```yaml
  train: ../train/images
  val: ../val/images

  nc: 2
  names: ['placard', 'text']
  ```
- `text_directory/`: Directory containing text files for different classes. For example, `animals.txt` might contain:
  ```txt
  cat
  dog
  elephant
  ```
- `best_placard.pt`: Trained YOLOv5 model weights for placard detection.
- `best_text_v3.pt`: Trained YOLOv5 model weights for text detection.

## Acknowledgements

This project was developed as a CSR initiative to aid cognitive skills development in children and the elderly through an interactive game mode.
