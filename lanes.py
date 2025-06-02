import cv2
import numpy as np

def make_points(image, line):
    if line is None or not isinstance(line, (list, tuple, np.ndarray)) or len(line) != 2:
        return None  # Return None if line is invalid
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1 * 3 / 5)  # slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None or len(lines) == 0:
        print("Debug: No lines detected.")
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope, intercept = fit
                if slope < 0:  # Negative slope means left lane
                    left_fit.append((slope, intercept))
                else:  # Positive slope means right lane
                    right_fit.append((slope, intercept))
            except Exception as e:
                print(f"Error fitting line: {e}")
                continue

    # Handle empty fits
    left_fit_average = np.mean(left_fit, axis=0) if len(left_fit) > 0 else None
    right_fit_average = np.mean(right_fit, axis=0) if len(right_fit) > 0 else None

    # Debugging: Print averages
    print(f"Debug: left_fit_average = {left_fit_average}")
    print(f"Debug: right_fit_average = {right_fit_average}")

    # Make points only if averages are valid
    left_line = make_points(image, left_fit_average) if left_fit_average is not None else None
    right_line = make_points(image, right_fit_average) if right_fit_average is not None else None

    return [line for line in [left_line, right_line] if line is not None]

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is None:
                continue
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(canny):
    height, width = canny.shape
    mask = np.zeros_like(canny)

    triangle = np.array([[
        (200, height),
        (550, 250),
        (1100, height)
    ]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

# Main script
if __name__ == "__main__":
    cap = cv2.VideoCapture("test2.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Debug: No more frames to read.")
            break

        try:
            canny_image = canny(frame)
            cropped_canny = region_of_interest(canny_image)
            lines = cv2.HoughLinesP(
                cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5
            )
            averaged_lines = average_slope_intercept(frame, lines)
            line_image = display_lines(frame, averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
            cv2.imshow("result", combo_image)
        except Exception as e:
            print(f"Error processing frame: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
