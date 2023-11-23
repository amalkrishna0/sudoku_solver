import cv2
import numpy as np

def process_image(frame):
    #frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # thresholding
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Just cleaning the image using dilate
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # I need the largest area so finding the max contours
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    corners = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    #drawing contours in a blank image
    contour_image = np.zeros_like(frame)
    cv2.drawContours(contour_image, [corners], -1, (255, 255, 255), 2)


    #wrapping the sudoku into a new image
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [0, 450], [450, 450], [450, 0]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result_image = cv2.warpPerspective(frame, matrix, (450, 450))
    
    # Display the processed images
    cv2.imshow('Original Image', frame)
    cv2.imshow('Processed Image', result_image)

# Open the camera
cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()

    cv2.imshow('Capture, Process, and Find Contour Corners', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):#when c is pressed it captures
        process_image(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#exits when  is pressed
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
