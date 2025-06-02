**🚗 Lane Detection Using OpenCV**
This project demonstrates a basic lane detection system using OpenCV and Python. It processes video frames from a car-mounted camera to identify and highlight road lane markings using computer vision techniques like Canny edge detection, Hough Transform, and brightness normalization.

**🧠 Features**
* Brightness & contrast normalization using CLAHE
* Region of interest filtering
* Canny edge detection
* Hough Line Transform for lane detection
* Line averaging for smoother detection
* Real-time video frame processing
* Visual overlay of detected lanes

**🛠️TechStack**
* Python 3.x
* OpenCV
* NumPy

**You can install the dependencies using:**
command :
pip install opencv-python numpy

**🧠 Algorithm Overview**
* adjust_brightness_contrast: Normalizes lighting using LAB color space and CLAHE.
* canny: Applies Canny edge detection with Gaussian blur.
* region_of_interest: Masks the frame to focus on the road region.
* average_slope_intercept: Computes average lane lines from detected segments.
* display_lines: Draws the detected lines on the original frame.
* HoughLinesP: Extracts line segments from the edge-detected image.

**ScreenShot**
![image](https://github.com/user-attachments/assets/48ad36b9-f88f-4a98-a0a1-f6ab1affa535)
