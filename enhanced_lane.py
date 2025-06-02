import cv2
import numpy as np

# Helper function to normalize brightness
def adjust_brightness_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Function to create points for a line
def make_points(image, line):
    if line is None or not isinstance(line, (list, tuple, np.ndarray)) or len(line) != 2:
        return None
    slope, intercept = line
    y1 = int(image.shape[0])  # Bottom of the image
    y2 = int(y1 * 3 / 5)      # Slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

# Function to average slopes and intercepts
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None or len(lines) == 0:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope, intercept = fit
                if slope < 0:  # Left lane (negative slope)
                    left_fit.append((slope, intercept))
                else:  # Right lane (positive slope)
                    right_fit.append((slope, intercept))
            except Exception as e:
                continue

    left_fit_average = np.mean(left_fit, axis=0) if len(left_fit) > 0 else None
    right_fit_average = np.mean(right_fit, axis=0) if len(right_fit) > 0 else None
    left_line = make_points(image, left_fit_average) if left_fit_average is not None else None
    right_line = make_points(image, right_fit_average) if right_fit_average is not None else None
    return [line for line in [left_line, right_line] if line is not None]

# Function for edge detection
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# Function to display lines
def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is None:
                continue
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# Function to define region of interest
def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)
    triangle = np.array([[
        (width * 0.1, height),
        (width * 0.5, height * 0.6),
        (width * 0.9, height)
    ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

# Main script for video processing
if __name__ == "__main__":
    cap = cv2.VideoCapture("road_car_view.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Debug: No more frames to read.")
            break

        try:
            # Preprocess frame
            adjusted_frame = adjust_brightness_contrast(frame)
            canny_image = canny(adjusted_frame)
            cropped_canny = region_of_interest(canny_image)

            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(
                cropped_canny, 
                2, 
                np.pi / 180, 
                100, 
                np.array([]), 
                minLineLength=50, 
                maxLineGap=10
            )
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)

            # Combine original frame with lane lines
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

            # Display the output
            cv2.imshow("Lane Detection", combo_image)
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
