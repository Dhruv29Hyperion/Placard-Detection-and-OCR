import cv2
import time
import requests
import os
import shutil


# Function to clear directories
def clear_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


# Directories to save the frames
frames_dir = 'frames'
clear_directory(frames_dir)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam initialized. Starting video capture...")

# Set parameters
frame_rate = 5  # Frames per second
total_duration = 30  # Duration to capture video (seconds)
api_url = "http://127.0.0.1:8000/detect_placard/"  # Adjust API URL as needed
language = "eng"  # Example language
class_name = "Tree"  # Example class name
timer = 1

# Capture frames
start_time = time.time()
frame_count = 0

while time.time() - start_time < total_duration:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Save frames at the specified rate
    frame_path = os.path.join(frames_dir, f"frame_{frame_count}.tiff")
    cv2.imwrite(frame_path, frame)

    frame_count += 1
    time.sleep(1 / frame_rate)

    # Every second, send the last 10 frames to the API
    if frame_count % frame_rate == 0:
        # Collect the last 10 frames
        frames = [os.path.join(frames_dir, f"frame_{i}.tiff") for i in range(frame_count - frame_rate, frame_count)]

        # Prepare files for API request
        files = []
        for f in frames:
            with open(f, 'rb') as file:
                files.append(('images', (os.path.basename(f), file.read(), 'image/tiff')))

        # Send request to API (send the current second as the timer)
        response = requests.post(api_url, files=files,
                                 data={'timer': timer, 'language': language, 'class_name': class_name})

        # Handle API response
        if response.status_code == 200:
            result = response.json()
            print(f"Timer: {timer}")
            print(f"Placard Detected: {result['placard_detected']}")
            print(f"Time Left: {total_duration - (time.time() - start_time):.2f} seconds")
            if result['placard_detected'] and timer == 3:
                print(f"Detected Text: {result['text']}")
                # Terminate the while loop if placard is detected and timer is 3
                break
        else:
            print(f"API request failed with status code: {response.status_code}")

        if result['placard_detected']:
            timer += 1
        else:
            timer = 1

        # Clean up sent frames
        for file in frames:
            os.remove(file)

# Release the webcam
cap.release()

# Final cleanup
clear_directory(frames_dir)


