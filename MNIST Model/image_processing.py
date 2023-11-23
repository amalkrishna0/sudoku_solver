import cv2
import numpy as np

def process_image(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours and find corners
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the contour and corners on the frame
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

        for corner in approx:
            x, y = corner[0]
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Perform perspective transformation if we detect four corners
        if len(approx) == 4:
            # Define the destination points for the perspective transform
            dst_points = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

            # Convert the source points to the required format
            src_points = np.float32(approx)

            # Get the perspective transform matrix
            perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Apply the perspective transform
            warped = cv2.warpPerspective(frame, perspective_matrix, (300, 300))

            # Display the warped image
            cv2.imshow('Warped', warped)

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Display the frame with an option to capture the image
    cv2.imshow('Capture Image', frame)

    # Capture an image when 'c' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('c'):
        captured_image = frame.copy()  # Capture the current frame
        cv2.imshow('Captured Image', captured_image)

        # Process the captured image
        process_image(captured_image)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
