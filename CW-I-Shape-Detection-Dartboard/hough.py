import numpy as np
import cv2
import os
import sys
import argparse
import face

# LOADING THE IMAGE
parser = argparse.ArgumentParser(description='Convert RGB to GRAY')
# parser.add_argument('-name', '-n', type=str, default='images/coins2.png')
# args = parser.parse_args()

# ==================================================
def sobelEdge(input):
    # intialise the output using the input
    edgeOutputX = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
    edgeOutputY = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
    # create the Gaussian kernel in 1D
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    kernelY = kernelX.T
    # create a padded version of the input
    kernelRadiusX = round((kernelX.shape[0] - 1) / 2)
    kernelRadiusY = round((kernelX.shape[1] - 1) / 2)
    paddedInput = cv2.copyMakeBorder(input,
        kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
        cv2.BORDER_REPLICATE)
    # convolution with sobel 
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            patch = paddedInput[i:i+kernelX.shape[0], j:j+kernelX.shape[1]]
            edgeOutputX[i, j] = (np.multiply(patch, kernelX)).sum()
            edgeOutputY[i, j] = (np.multiply(patch, kernelY)).sum()
    return edgeOutputX, edgeOutputY

def is_circle_fully_in_box(circle, box_coords):
    cx, cy, r = circle
    x1, y1 = box_coords[0]
    x2, y2 = box_coords[1]
    return (cx - r >= x1) and (cx + r <= x2) and (cy - r >= y1) and (cy + r <= y2)

def circle_box_intersection_ratio(circle, box_coords):
    cx, cy, r = circle
    x1, y1 = box_coords[0]
    x2, y2 = box_coords[1]

    # Create a grid to approximate intersection area
    # Use bounding rectangle of circle for efficiency
    x_min = max(int(cx - r), x1)
    x_max = min(int(cx + r), x2)
    y_min = max(int(cy - r), y1)
    y_max = min(int(cy + r), y2)

    if x_max <= x_min or y_max <= y_min:
        return 0.0  # no overlap

    # Create meshgrid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    # Count points inside circle
    inside_circle = (xx - cx)**2 + (yy - cy)**2 <= r**2
    intersection_area = np.sum(inside_circle)
    circle_area = np.pi * r**2
    return intersection_area / circle_area

def hough_line_detector(box_coords):
    x1, y1 = box_coords[0]
    x2, y2 = box_coords[1]
    roi = image[y1:y2, x1:x2]

    # Edge detection in the ROI
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)  # adjust threshold

    # Draw lines on the original image
    if lines is not None:
        for rho_theta in lines:
            rho, theta = rho_theta[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # Line endpoints relative to ROI
            x1 = int(x0 + 1000*(-b)) + x
            y1 = int(y0 + 1000*(a)) + y
            x2 = int(x0 - 1000*(-b)) + x
            y2 = int(y0 - 1000*(a)) + y
            cv2.line(imagewithcircle, (x1, y1), (x2, y2), (0, 0, 255), 2)

# ==== MAIN ==============================================

imageName = "Dartboard/dart0.jpg"

# ignore if no such file is present.
if not os.path.isfile(imageName):
    print('No such file')
    sys.exit(1)

# Read image from file
image = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(image) is np.ndarray):
    print('Not image data')
    sys.exit(1)


boxes = face.guesses
print(boxes[0])

# CONVERT COLOUR, BLUR AND SAVE
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = gray_image.astype(np.float32)

# apply sobel
edgemapX, edgemapY = sobelEdge(gray_image)
# magnitude
magnitude = np.sqrt(edgemapX**2 + edgemapY**2)
normmagnitude = (magnitude-magnitude.min())/(magnitude.max()-magnitude.min())
# orientation
anglemap = np.arctan2(edgemapY, edgemapX)
# edge
edgemap = normmagnitude > 0.2

rmin = 20
rmax = 100
# create hough space
hough3D = np.zeros([image.shape[0], image.shape[1], rmax-rmin+1], dtype=np.float32)

for i in range(0, image.shape[0]):  # go through all rows (or scanlines)
    for j in range(0, image.shape[1]):
        # if pixel is an edge (>thr)
        if edgemap[i, j] > 0:
            for r in range(rmin, rmax+1):
                x = (j + np.array([-1, 1])*r*np.cos(anglemap[i, j])).astype(int)
                y = (i + np.array([-1, 1])*r*np.sin(anglemap[i, j])).astype(int)
                for k in range(0, 2):
                    if (y[k] >= 0) and (y[k] < image.shape[0]) and (x[k] >= 0) and (x[k] < image.shape[1]):
                        # accumulate vote for circle centre
                        hough3D[y[k], x[k], r-rmin] += 1


hough2D = np.sum(hough3D, axis=2)

threshold = 20  # Try to change the threshold
circle_parameters_ls = []
for i in range(0, hough3D.shape[0]):
    for j in range(0, hough3D.shape[1]):
        for k in range(0, hough3D.shape[2]):
            if hough3D[i, j, k] >= threshold:
                circle_parameters_ls.append([j, i, k + rmin])
# Plot the circles according to the parameters on the original image
imagewithcircle = image
for circle_parameters in circle_parameters_ls:
    imagewithcircle = cv2.circle(imagewithcircle,
                                 (circle_parameters[0], circle_parameters[1]),
                                 int(circle_parameters[2]),
                                 color=(0, 0, 255),
                                 thickness=2)

boxes = face.guesses
THR = 60

# for box in boxes:
#     for idx, circle in enumerate(circle_parameters_ls):
#         if is_circle_fully_in_box(circle, box) or circle_box_intersection_ratio(circle, box) >= THR:
#             # apply line detector
#             hough_line_detector(box)

# -------------------------- change so that you're only checking within the boxes


cv2.imwrite("imagewithcircle.jpg", imagewithcircle)
# # save image
# cv2.imwrite("edgemapX.jpg", (edgemapX-edgemapX.min())/(edgemapX.max()-edgemapX.min())*255)
# cv2.imwrite("edgemapY.jpg", (edgemapY-edgemapY.min())/(edgemapY.max()-edgemapY.min())*255)
# cv2.imwrite("hough2D.jpg", (hough2D-hough2D.min())/(hough2D.max()-hough2D.min())*255)