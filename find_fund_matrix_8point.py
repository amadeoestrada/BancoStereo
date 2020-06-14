#!/usr/bin/env python

"""
    This Python program finds the fundamental matrix using two images of the same scenario.
    A check board pattern and a stereo setup are used. The match coordinates of the corners
    of the check board pattern are found using HarrisCorners and CornersSubPix corrections.

    Then the program uses the 8 point algorithm to find the F matrix.
    It uses the tool computeCorrespondEpilines.

    Finally, the program draws the epipolar lines and the match coordinates in the original
    images.
"""
__author__ = "Amadeo Estrada"
__date__ = "13 / June / 2020"

import numpy as np
import cv2
import glob
import sys
from matplotlib import pyplot as plt

# ---------------------- PARAMETERS SET
nRows = 9
nCols = 6
dimension = 25  # - mm

# Define the calibration folder
workingFolder = "./calibracion"
imageType = 'png'

# Change the resolution according to the images used
image_res_x = 1920  # input image horizontal resolution
image_res_y = 1080  # input image vertical resolution
# ------------------------------------------

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nRows * nCols, 3), np.float32)
objp[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints_r = []  # 2d points in image plane. RIGHT
imgpoints_l = []  # 2d points in image plane. LEFT

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Process the RIGHT camera file
filename = workingFolder + "/right/*." + imageType
images = glob.glob(filename)

print(len(images))
if len(images) > 1:
    print("More than one RIGHT image found. ABORT!")
    sys.exit()

else:
    for fname in images:
        # -- Read the file and convert in greyscale
        img_r = cv2.imread(fname)
        # Clone the original image to keep original unchanged
        img2 = img_r.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        print("Reading image ", fname)

        # Find the chess board corners
        ret_r, corners_r = cv2.findChessboardCorners(gray, (nCols, nRows), None)

        # If found, add object points, image points (after refining them)
        if ret_r:
            print("Pattern found! Press ESC to skip or ENTER to accept")
            # --- Sometimes, Harris corners fails with crappy pictures, so
            corners2_r = cv2.cornerSubPix(gray, corners_r, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(img_r, (nCols, nRows), corners2_r, ret_r)
            cv2.imshow('Right Camera', img_r)

            # User checks image quality
            k = cv2.waitKey(0) & 0xFF
            if k == 27:  # -- ESC Button
                print("Image is not good. USER ABORTED!")
                sys.exit()
            # Notify that image was accepted
            print("Right image accepted")
            # Save the the detected coordinates
            imgpoints_r.append(corners2_r)

        else:
            print("Pattern not found. ABORT!")
            sys.exit()

cv2.destroyAllWindows()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Process the LEFT camera file
filename = workingFolder + "/left/*." + imageType
images = glob.glob(filename)

print(len(images))
if len(images) > 1:
    print("More than one LEFT image found. ABORT!")
    sys.exit()

else:
    for fname in images:
        # -- Read the file and convert in greyscale
        img_l = cv2.imread(fname)
        # Clone the original image to keep original unchanged
        img1 = img_l.copy()
        # Convert to grayscale
        gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        print("Reading image ", fname)

        # Find the chess board corners
        ret_l, corners_l = cv2.findChessboardCorners(gray, (nCols, nRows), None)

        # If found, add object points, image points (after refining them)
        if ret_l:
            print("Pattern found! Press ESC to skip or ENTER to accept")
            # --- Sometimes, Harris corners fails with crappy pictures, so
            corners2_l = cv2.cornerSubPix(gray, corners_l, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(img_l, (nCols, nRows), corners2_l, ret_l)
            cv2.imshow('Right Camera', img_l)

            # User checks image quality
            k = cv2.waitKey(0) & 0xFF
            if k == 27:  # -- ESC Button
                print("Image is not good. USER ABORTED!")
                sys.exit()
            # notify that image was accepted
            print("Left image accepted")
            # Save the the detected coordinates
            imgpoints_l.append(corners2_l)
        else:
            print("Patter not found. ABORT!")
            sys.exit()

cv2.destroyAllWindows()

# - - - - - - - - - Extract coordinates part - - - - - - - - - - - - -

# Extract the coordinates from the image points list (3 dimension array)
right_coord = np.array(imgpoints_r[0])  # convert from list to array
right_coord = np.squeeze(right_coord)  # Convert to 2 dimension array
# Vertically flip array. Checker board pattern results are ordered
# from-bottom right to top-left
coord_right = np.flipud(right_coord)

# Extract the coordinates from the image points list (3 dimension array)
left_coord = np.array(imgpoints_l[0])  # convert from list to array
left_coord = np.squeeze(left_coord)  # Convert to 2 dimension array
# Vertically flip array. Checker board pattern results are ordered
# from-bottom right to top-left
coord_left = np.flipud(left_coord)

# - - - - - - -   Add third dimension with ones - - - - - - - - -

# Add z coordinates with ones
z_coord = np.ones((54, 1), np.float32)
# Now, join z coordinates to the left of RIGHT coordinates
right_coord = np.concatenate((coord_right, z_coord), axis=1)
left_coord = np.concatenate((coord_left, z_coord), axis=1)

# Reshape arrays to 3 * 54
coord_right = right_coord.T
coord_left = left_coord.T

# - - - - - - -   Fundamental matrix routines  - - - - - - - - -

def compute_fundamental(x1x,x2x):
    """ Computes the fundamental matrix from corresponding points
                (x1x,x2x 3*n arrays) using the 8 point algorithm.
                Each row in the Ax matrix below is constructed as
                [x*x', y*x', x', x*y', y*y', y, x', y', 1]
    """
    n = x1x.shape[1]
    if x2x.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    Ax = np.zeros((n,9))
    indice = 0

    # [x*x', y*x', x', x*y', y*y', y, x', y', 1]
    while indice < n:
        Ax[indice][0] = x1x[0][indice] * x2x[0][indice]  # x*x'
        Ax[indice][1] = x1x[0][indice] * x2x[1][indice]  # y*x'
        Ax[indice][2] = x1x[0][indice]                   # x'
        Ax[indice][3] = x1x[1][indice] * x2x[0][indice]  # x*y'
        Ax[indice][4] = x1x[1][indice] * x2x[1][indice]  # y*y'
        Ax[indice][5] = x1x[1][indice]                   # y
        Ax[indice][6] = x2x[0][indice]                   # x'
        Ax[indice][7] = x2x[1][indice]                   # y'
        Ax[indice][8] = 1                                # 1
        indice += 1

    # compute linear least square solution
    U,S,V = np.linalg.svd(Ax)
    Fx2 = V[-1].reshape(3,3)

    # constrain Fx2
    # make rank 2 by zeroing out last singular value
    U,S,V = np.linalg.svd(Fx2)
    S[2] = 0
    Fx2 = np.dot(U,np.dot(np.diag(S),V))
    return Fx2/Fx2[2,2]

def fundamental_matrix(x1x,x2x):
    n = x1x.shape[1]
    if x2x.shape[1] != n:
        raise ValueError("Number of points don't match.")
    # normalize image coordinates
    x1x = x1x / x1x[2]
    mean_1 = np.mean(x1x[:2],axis=1)
    S1 = np.sqrt(2) / np.std(x1x[:2])
    T1 = np.array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
    x1x = np.dot(T1,x1x)
    x2x = x2x / x2x[2]
    mean_2 = np.mean(x2x[:2],axis=1)
    S2 = np.sqrt(2) / np.std(x2x[:2])
    T2 = np.array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
    x2x = np.dot(T2,x2x)
    # compute Fx with the normalized coordinates
    Fx = compute_fundamental(x1x,x2x)
    # reverse normalization
    Fx = np.dot(T1.T,np.dot(Fx,T2))

    return Fx/Fx[2,2]

# - - - - - - - - -  Fundamental Matrix call out  - - - - - - - - - - - - -

F = fundamental_matrix(coord_left, coord_right)

coord_right = coord_right.T
coord_left = coord_left.T

coord_left, no_use = np.array_split(coord_left, [2], axis=1)
coord_right, no_use = np.array_split(coord_right, [2], axis=1)

# - - - - - - - -  Draw epipolar lines - - - - - - - - - - - - -
# Draw epipolar images using F calculated with 8 point algorithm

# Function to draw epipolar lines and match circles to images.
def drawlines(img_a, img_b, lines, pts1, pts2):
    """ img_a - image on which we draw the epilines for the points in img_b lines
        - corresponding epilines.
        The lines argument contains an equation for each epipolar line. The for
        cycle below takes the iterables lines, pts1, and pts2 into a tuple assigned
        to r[0], r[1], r[2], pt1, pt2, for each for iteration.
        Therefore, each iteration will draw a line from the extreme right (x = 0)
        to the extreme left (x = c) and the correspondent coordinates of the
        matching points on the two images.
    """
    # Assign row, column and color information
    r, c, color_info = img_a.shape

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        # Use random color for each epipolar line
        color = tuple(np.random.randint(0, 255, 3).tolist())
        # Set the start of line to the extreme left of the image
        x0, y0 = map(int, [0, -r[2] / r[1]])
        # Set the end of the line to the extreme right of the image
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        # Draw the line using the coordinates above
        img_a = cv2.line(img_a, (x0, y0), (x1, y1), color, 1)
        # Draw the matching coordinates for each image
        img_a = cv2.circle(img_a, tuple(pt1), 5, color, -1)
        img_b = cv2.circle(img_b, tuple(pt2), 5, color, -1)
    return img_a, img_b

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(coord_left.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img2, img1, lines1, coord_right, coord_left)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(coord_right.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img1, img2, lines2, coord_left, coord_right)

# Show plot of the two epipolar line images
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

# Save results in results folder
cv2.imwrite(workingFolder + "/results/img1_8point.png", img1)
cv2.imwrite(workingFolder + "/results/img2_8point.png", img2)

# - - - - - - - - - END of the program - - - - - - - - - - - - -