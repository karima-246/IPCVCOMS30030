import numpy as np
import cv2
import os
import sys
import argparse

# from skimage import data, color, img_as_ubyte
# from skimage.feature import canny
# from skimage.transform import hough_ellipse
# from skimage.draw import ellipse_perimeter


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

def get_edge_map_dir(frame_gray):
    # apply sobel
    edgemapX, edgemapY = sobelEdge(frame_gray)
    # magnitude
    magnitude = np.sqrt(edgemapX**2 + edgemapY**2)
    normmagnitude = (magnitude-magnitude.min())/(magnitude.max()-magnitude.min())
    # orientation
    anglemap = np.arctan2(edgemapY, edgemapX)
    # edge
    edgemap = normmagnitude > 0.2

    return anglemap, edgemap

def line_detect(box_coords, edgemap, image,
                theta_res=np.deg2rad(1),
                rho_res=1):

    (x1, y1), (x2, y2) = box_coords

    # Extract ROI
    roi = edgemap[y1:y2, x1:x2]
    H, W = roi.shape

    # Hough parameter ranges
    diag_len = int(np.ceil(np.sqrt(H*H + W*W)))
    rhos = np.arange(-diag_len, diag_len + 1, rho_res)
    thetas = np.arange(0, np.pi, theta_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.float32)

    # --- MAIN VOTING LOOP ---
    for y in range(H):
        for x in range(W):
            if roi[y, x] > 0:        # edge pixel
                for ti, theta in enumerate(thetas):
                    # local rho
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    ri = int((rho + diag_len) / rho_res)
                    accumulator[ri, ti] += 1

    # --- EXTRACT PEAKS ---
    lines = []
    for ri in range(len(rhos)):
        for ti in range(len(thetas)):
            if accumulator[ri, ti] >= threshold_l:
                rho_local = rhos[ri]
                theta = thetas[ti]

                # Convert from ROI-local to global rho:
                # ρ_global = ρ_local + (x1*cosθ + y1*sinθ)
                rho_global = rho_local + x1*np.cos(theta) + y1*np.sin(theta)

                lines.append((rho_global, theta))

    # --- DRAW LINES ---
    imagewithline = image
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1_line = int(x0 + 2000 * (-b))
        y1_line = int(y0 + 2000 * (a))
        x2_line = int(x0 - 2000 * (-b))
        y2_line = int(y0 - 2000 * (a))

        imagewithline = cv2.line(imagewithline, (x1_line, y1_line), (x2_line, y2_line), color=(255, 0, 0), thickness=2)
    
    lines_found = False

    if len(lines) > 0:
        lines_found = True

    return lines_found, imagewithline


def circle_detect(box_coords, edgemap, anglemap, image):
    (x1, y1), (x2, y2) = box_coords
    # Extract ROI
    roi_edge = edgemap[y1:y2, x1:x2]
    roi_angle = anglemap[y1:y2, x1:x2]
    # Dimensions of ROI
    H = y2 - y1
    W = x2 - x1

    # Create local Hough accumulator for ROI only
    hough_local = np.zeros([H, W, rmax - rmin + 1], dtype=np.float32)
    # Hough voting inside ROI
    for i in range(H):        # y inside ROI
        for j in range(W):    # x inside ROI
            if roi_edge[i, j] > 0:  # edge pixel
                theta = roi_angle[i, j]
                for r in range(rmin, rmax + 1):
                    # potential circle centres (local coords)
                    x_c = (j + np.array([-1, 1]) * r * np.cos(theta)).astype(int)
                    y_c = (i + np.array([-1, 1]) * r * np.sin(theta)).astype(int)

                    for k in range(2):
                        if (0 <= x_c[k] < W) and (0 <= y_c[k] < H):
                            hough_local[y_c[k], x_c[k], r - rmin] += 1
    # Threshold local Hough accumulator
    circles_found = False
    circle_parameters_ls = []
    for i in range(H):
        for j in range(W):
            for k in range(rmax - rmin + 1):
                if hough_local[i, j, k] >= threshold_c:
                    # Convert back to GLOBAL image coordinates
                    cx = j + x1
                    cy = i + y1
                    radius = k + rmin
                    circle_parameters_ls.append([cx, cy, radius])
                    circles_found = True
    
    imagewithcircle = image
    for circle_parameters in circle_parameters_ls:
        imagewithcircle = cv2.circle(imagewithcircle,
                                    (circle_parameters[0], circle_parameters[1]),
                                    int(circle_parameters[2]),
                                    color=(255, 0, 0),
                                    thickness=2)
    return circles_found, imagewithcircle


# ==== MAIN ==============================================

# parameters

threshold_c = 15
threshold_l = 110

rmin = 15
rmax = 100




# cv2.imwrite("imagewithcircle.jpg", imagewithcircle)
# # save image
# cv2.imwrite("edgemapX.jpg", (edgemapX-edgemapX.min())/(edgemapX.max()-edgemapX.min())*255)
# cv2.imwrite("edgemapY.jpg", (edgemapY-edgemapY.min())/(edgemapY.max()-edgemapY.min())*255)
# cv2.imwrite("hough2D.jpg", (hough2D-hough2D.min())/(hough2D.max()-hough2D.min())*255)