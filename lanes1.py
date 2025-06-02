import cv2
import numpy as np

# Function to create points from slope and intercept
def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])  # Bottom of the image
    y2 = int(y1 * 3 / 5)      # Slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

# Function to calculate average slope and intercept for left and right lanes
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)  # Fit a line (y = mx + b)
            slope, intercept = fit
            if slope < 0:  # Negative slope indicates left lane
                left_fit.append((slope, intercept))
            else:  # Positive slope indicates right lane
                right_fit.append((slope, intercept))
    # Calculate averages
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    left_line = make_points(image, left_fit_average) if left_fit_average is not None else None
    right_line = make_points(image, right_fit_average) if right_fit_average is not None else None
    return [line for line in [left_line, right_line] if line is not None]

# Function to apply Canny edge detection
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# Function to display lines on an image
def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # Blue lines with thickness 10
    return line_image

# Function to create a region of interest mask
def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)
    # Define a triangular region of interest
    triangle = np.array([[
        (200, height),
        (550, 250),
        (1100, height)
    ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)  # Fill triangle with white
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

# Main code to process a single image
image = cv2.imread('test_image.jpg')  # Load the image
if image is None:
    print("Error: Image not found. Check the file path.")
else:
    lane_image = np.copy(image)  # Make a copy to avoid modifying the original image
    lane_canny = canny(lane_image)  # Apply Canny edge detection
    cropped_canny = region_of_interest(lane_canny)  # Apply region of interest mask
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(image, lines)  # Average the detected lines
    line_image = display_lines(lane_image, averaged_lines)  # Create an image with lane lines
    # Combine the original image with the lane line image
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)

    # Display the output
    cv2.imshow("Lane Detection", combo_image)
    print("Press any key to close the window...")
    cv2.waitKey(0)  # Wait indefinitely for a key press
    cv2.destroyAllWindows()  # Close all OpenCV windows
