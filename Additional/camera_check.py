import cv2
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam initialized. Capturing a frame after 2 seconds...")

    # Wait for 2 seconds
    time.sleep(2)

    # Capture a frame
    ret, frame = cap.read()

    if ret:
        # Display the captured frame
        cv2.imshow('Sample Frame', frame)
        cv2.imwrite('sample_frame.jpg', frame)
        print("Frame captured and saved as 'sample_frame.jpg'. Press any key to close the window.")

        # Wait for a key press and then close the image window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not capture frame.")

# Release the webcam
cap.release()
print("Webcam released.")
