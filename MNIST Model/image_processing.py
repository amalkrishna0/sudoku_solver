import cv2
import numpy as np

def process_image(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Apply morphology (closing operation)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=2)
    # Dilate the image

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    

    contour_image = np.zeros_like(frame)
    cv2.drawContours(contour_image, [corners], -1, (255, 255, 255), 2)

    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [0, 450], [450, 450], [450, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result_image = cv2.warpPerspective(frame, matrix, (450, 450))
    
    # Display the processed images
    cv2.imshow('Original Image', frame)
    cv2.imshow('Processed Image', result_image)

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Display the frame with an option to process the image
    cv2.imshow('Capture, Process, and Find Contour Corners', frame)

    # Process the captured image when 'c' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        process_image(frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
