import cv2
import numpy as np
from image_extraction import process_image

cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()

    cv2.imshow('Capture, Process, and Find Contour Corners', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        result_image = process_image(frame)

        # Further processing using the result_image in this file
        gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(thresholded)

        cv2.imshow('Inverted Image', inverted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
