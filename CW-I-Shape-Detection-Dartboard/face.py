################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

import numpy as np
import cv2
import os
import sys
import argparse

# LOADING THE IMAGE
# Example usage: python filter2d.py -n car1.png
parser = argparse.ArgumentParser(description='face detection')
# parser.add_argument('-name', '-n', type=str, default='Dartboard/dart0.jpg')
args = parser.parse_args()

# /** Global variables */
cascade_name = "../CW-I-Shape-Detection-Dartboard/Dartboardcascade/cascade.xml"


def detectAndDisplay(frame):
	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.128, minNeighbors=4, flags=0, minSize=(50,50), maxSize=(200,200))
    # 3. Print number of Faces found
    # print(len(faces))
    # 4. Draw box around faces found
    for i in range(0, len(faces)):
        start_point = (faces[i][0], faces[i][1])
        end_point = (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)
        guesses.append(np.array([start_point, end_point]))
    


def testPerformance():
    THRESHOLD = 0.5
    
    ground_truth = ground_truths[imageName]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Track which guesses have been matched
    guess_matched = [False] * len(guesses)

    for i in range(len(ground_truth)):
        x = int(ground_truth[i][0])
        y = int(ground_truth[i][1])
        width = int(ground_truth[i][2])
        height = int(ground_truth[i][3])

        gt_x1, gt_y1 = x, y
        gt_x2, gt_y2 = x + width, y + height

        matched_gt = False

        for j, guess in enumerate(guesses):
            guess_x1 = guess[0][0]
            guess_y1 = guess[0][1]
            guess_x2 = guess[1][0]
            guess_y2 = guess[1][1]

            # overlap
            x_overlap = max(0, min(gt_x2, guess_x2) - max(gt_x1, guess_x1))
            y_overlap = max(0, min(gt_y2, guess_y2) - max(gt_y1, guess_y1))
            overlap_area = x_overlap * y_overlap

            # areas
            gt_area = width * height
            guess_area = (guess_x2 - guess_x1) * (guess_y2 - guess_y1)
            union_area = gt_area + guess_area - overlap_area
            iou = overlap_area / union_area if union_area > 0 else 0

            # TP if above threshold
            if iou >= THRESHOLD:
                if not guess_matched[j]:
                    true_positives += 1
                    guess_matched[j] = True
                matched_gt = True

        # FN = this ground truth was never matched
        if not matched_gt:
            false_negatives += 1

    # FP = guesses that weren’t matched to any GT
    for matched in guess_matched:
        if not matched:
            false_positives += 1

    # ---- METRICS ----
    # Precision = TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    # TPR / Recall = TP / (TP + FN)
    tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # F1 Score = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

    return tpr, f1


def detectAndDisplayGroundTruth(frame, imageName):
    ground_truth = ground_truths[imageName]
    for i in range (len(ground_truth)):
        x = int(ground_truth[i][0])
        y = int(ground_truth[i][1])
        width = int(ground_truth[i][2])
        height = int(ground_truth[i][3])

        # Draw box around ground truth
        start_point = (x, y)
        end_point = (x + width, y + height)
        colour = (0, 0, 255)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

def readGroundTruth(filename='groundtruth.txt'):
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            x = float(content_list[1])
            y = float(content_list[2])
            width = float(content_list[3])
            height = float(content_list[4])
            # print(str(x)+' '+str(y)+' '+str(width)+' '+str(height))
            ground_truths.setdefault(img_name, []).append(np.array([x, y, width, height]))



# ==== MAIN ==============================================

ground_truths = {}
tprs = []
f1s = []
tpr_sum = 0
f1_sum = 0

readGroundTruth()

for i in range (16):
    imagePath = "Dartboard/dart" + str(i) + ".jpg"
    imageName = f"dart{i}.jpg"

    # ignore if no such file is present.
    if (not os.path.isfile(imagePath)):
        print('No such file image')
        sys.exit(1)

    if (not os.path.isfile(cascade_name)):
        print('No such file cascade')
        sys.exit(1)

    # 1. Read Input Image
    frame = cv2.imread(imagePath, 1)

    # ignore if image is not array.
    if not (type(frame) is np.ndarray):
        print('Not image data')
        sys.exit(1)


    # 2. Load the Strong Classifier in a structure called `Cascade'
    model = cv2.CascadeClassifier()
    if not model.load(cascade_name): # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
        print('--(!)Error loading cascade model')
        exit(0)


    # 3. Detect Faces and Display Result
    guesses = []

    tp = 0
    fp = 0
    fn = 0
    
    detectAndDisplay( frame )

    detectAndDisplayGroundTruth(frame, imageName)
    
    tpr, f1 = testPerformance()
    
    tprs.append(tpr)
    f1s.append(f1)

    tpr_sum = tpr_sum + tpr
    f1_sum = f1_sum + f1

    # 4. Save Result Image
    cv2.imwrite(f"detected/detected{i}.jpg", frame )

tpr_avg = tpr_sum / 16
f1_avg = f1_sum / 16

