#!/usr/bin/env python

"""
    This Python program finds the fundamental matrix using two images of the same scenario.
    A checkboard pattern and a stereo setup are used.
"""
__author__ = "Amadeo Estrada"
__date__ = "14 / May / 2020"

import numpy as np
import cv2
import glob
import sys
from matplotlib import pyplot as plt
import argparse

# ---------------------- SET THE PARAMETERS
nRows = 9
nCols = 6
dimension = 25  # - mm

workingFolder = "./calibracion"
imageType = 'png'

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

if len(sys.argv) < 6:
    print("\n Not enough inputs are provided. Using the default values.\n\n" \
          " type -h for help")
else:
    workingFolder = sys.argv[1]
    imageType = sys.argv[2]
    nRows = int(sys.argv[3])
    nCols = int(sys.argv[4])
    dimension = float(sys.argv[5])

if '-h' in sys.argv or '--h' in sys.argv:
    print("\n IMAGE CALIBRATION GIVEN A SET OF IMAGES")
    print(" call: python cameracalib.py <folder> <image type> <num rows (9)> <num cols (6)> <cell dimension (25)>")
    print("\n The script will look for every image in the provided folder and will show the pattern found." \
          " User can skip the image pressing ESC or accepting the image with RETURN. " \
          " At the end the end the following files are created:" \
          "  - cameraDistortion.txt" \
          "  - cameraMatrix.txt \n\n")

    sys.exit()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Process the RIGHT camera file
filename = workingFolder + "/right/*." + imageType
images = glob.glob(filename)

print(len(images))
if len(images) > 1:
    print("More than one RIGHT image found. ABORT!")
    sys.exit()

else:
    # nPatternFound = 0
    # imgNotGood = images[1]

    for fname in images:
        # if 'calibresult' in fname: continue
        # -- Read the file and convert in greyscale
        img_r = cv2.imread(fname)
        img1 = img_r
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        print("Reading image ", fname)

        # Find the chess board corners
        ret_r, corners_r = cv2.findChessboardCorners(gray, (nCols, nRows), None)

        # If found, add object points, image points (after refining them)
        if ret_r == True:
            print("Pattern found! Press ESC to skip or ENTER to accept")
            # --- Sometimes, Harris cornes fails with crappy pictures, so
            corners2_r = cv2.cornerSubPix(gray, corners_r, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(img_r, (nCols, nRows), corners2_r, ret_r)
            cv2.imshow('Right Camera', img_r)
            # cv2.waitKey(0)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:  # -- ESC Button
                print("Image is not good. USER ABORTED!")
                # imgNotGood = fname
                sys.exit()

            print("Right image accepted")
            # nPatternFound += 1
            # objpoints.append(objp)
            imgpoints_r.append(corners2_r)

            # cv2.waitKey(0)
        else:
            print("Patter not found. ABORT!")
            # imgNotGood = fname
            sys.exit()

cv2.destroyAllWindows()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Process the LEFT camera file
filename = workingFolder + "/left/*." + imageType
images = glob.glob(filename)

print(len(images))
if len(images) > 1:
    print("More than one LEFT image found. ABORT!")
    sys.exit()

else:
    # nPatternFound = 0
    # imgNotGood = images[1]

    for fname in images:
        # if 'calibresult' in fname: continue
        # -- Read the file and convert in greyscale
        img_l = cv2.imread(fname)
        img2 = img_l
        gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

        print("Reading image ", fname)

        # Find the chess board corners
        ret_l, corners_l = cv2.findChessboardCorners(gray, (nCols, nRows), None)

        # If found, add object points, image points (after refining them)
        if ret_l == True:
            print("Pattern found! Press ESC to skip or ENTER to accept")
            # --- Sometimes, Harris cornes fails with crappy pictures, so
            corners2_l = cv2.cornerSubPix(gray, corners_l, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(img_l, (nCols, nRows), corners2_l, ret_l)
            cv2.imshow('Right Camera', img_l)
            # cv2.waitKey(0)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:  # -- ESC Button
                print("Image is not good. USER ABORTED!")
                # imgNotGood = fname
                sys.exit()

            print("Left image accepted")
            # nPatternFound += 1
            # objpoints.append(objp)
            imgpoints_l.append(corners2_l)

            # cv2.waitKey(0)
        else:
            print("Patter not found. ABORT!")
            # imgNotGood = fname
            sys.exit()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Convert the pixel coordinates into normalised coordinates (-1 to 1) with the centroid ( 0 ) at the center
# of the image.

# --- Normalise RIGHT image coordinates

# Extract the coordinates from the image points list (3 dimension array)
right_coord = np.array(imgpoints_r[0])  # convert from list to array
right_coord = np.squeeze(right_coord)  # Convert to 2 dimension array
#x_coord, y_coord = np.array_split(right_coord, 2, 1)  # Split array into x_coord and y_coord

# Normalize coordinates
#x_coord = x_coord / (image_res_x / 2) - 1  # Normalise x_coord
#y_coord = y_coord / (image_res_y / 2) - 1  # Normalise y_coord

# Create Z column with ones
z_coord = np.ones((54,1), np.float32)

# Join x and y coordinates into RIGHT coordinates
#right_coord = np.concatenate((x_coord, y_coord), axis=1)
# Now, join z coordinates to the left of RIGHT coordinates
right_coord = np.concatenate((right_coord, z_coord), axis=1)

# --- Normalise LEFT image coordinates

# Extract the coordinates from the image points list (3 dimension array)
left_coord = np.array(imgpoints_l[0])  # convert from list to array
left_coord = np.squeeze(left_coord)  # Convert to 2 dimension array
#x_coord, y_coord = np.array_split(left_coord, 2, 1)  # Split array into x_coord and y_coord

# Normalize coordinates
#x_coord = x_coord / (image_res_x / 2) - 1  # Normalise x_coord
#y_coord = y_coord / (image_res_y / 2) - 1  # Normalise y_coord

# Join x and y coordinates into RIGHT coordinates
#left_coord = np.concatenate((x_coord, y_coord), axis=1)
# Now, join z coordinates to the left of LEFT coordinates
left_coord = np.concatenate((left_coord, z_coord), axis=1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Find the fundamental matrix and use OpenCV routine for comparison
F1, mask = cv2.findFundamentalMat(right_coord,left_coord,cv2.FM_8POINT)

# We select only inlier points
right_coord = right_coord[mask.ravel()==1]
left_coord = left_coord[mask.ravel()==1]

def drawlines(img1,img2,lines,right_coord,left_coord):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c,v1 = img1.shape
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,right_coord,left_coord):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        #img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        #img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

#Copy original coordinates.
right_coord_x = right_coord
left_coord_x = left_coord

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(left_coord.reshape(-1,1,2), 2,F1)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,right_coord,left_coord)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(right_coord.reshape(-1,1,2), 1,F1)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,left_coord,right_coord)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()


right_coord = right_coord.T         # transpose from 54 x 3 to 3 x 54
left_coord = left_coord.T           # transpose from 54 x 3 to 3 x 54

# Create the normalisation matrix
norm = np.array([[1.041666667E-03, 0, -1],[0, 1.851851852E-03, -1],[0, 0, 1]])
# Normalise using the normalisation matrix (RIGHT), rename as x1

x1 = np.matmul(norm,right_coord)
# Normalise using the normalisation matrix (LEFT), rename as x2

x2 = np.matmul(norm,left_coord)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create matrix A from the correlated points of left and right coordinates

A = np.zeros((54, 9), np.float32)
indice = 0

while indice < 54:
    A[indice][0] = x1[0][indice] * x2[0][indice]
    A[indice][1] = x1[1][indice] * x2[0][indice]
    A[indice][2] = x2[0][indice]
    A[indice][3] = x1[0][indice] * x2[1][indice]
    A[indice][4] = x1[1][indice] * x2[1][indice]
    A[indice][5] = x2[1][indice]
    A[indice][6] = x1[0][indice]
    A[indice][7] = x1[1][indice]
    A[indice][8] = 1
    indice += 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Calculate SVD
#A = A.T
u, s, vh = np.linalg.svd(A, full_matrices=True)

# extract the last column of matrix vh
no_use, F = np.array_split(vh, [8], axis=1)

# re-arrange Fundamental Matrix
F = np.reshape(F, (3, 3))

F = F.T

u, s, vh = np.linalg.svd(F, full_matrices=True)

D = np.zeros((3, 3), np.float32)
D[0][0] = s[0]
D[1][1] = s[1]

F = np.matmul(u,D)
F = np.matmul(F,vh.T)

#no_use, F = np.array_split(vh, [2], axis=1)
#D = np.zeros((3, 3), np.float32)
#D[0][0] = F[0]
#D[1][1] = F[1]
#D[0][0] = s[0][0]
#fund_mat = np.matmul(u, D)
#fund_mat = np.matmul(fund_mat, vh)

# De-normalise
F = np.matmul(norm.T,F)
F = np.matmul(F,norm)



# test
#test_1 = np.array([.16066337, .42508912, 1])
#test_2 = np.array([[-.25848138], [.2863686], [1]])

test_1 = np.array([380, 245, 1])
test_2 = np.array([[927], [429], [1]])

result = np.matmul(test_1, F1)
result2 = np.reshape(result,(1,3))
result2 = np.linalg.pinv(result2)
result = np.matmul(result, test_2)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(left_coord_x.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,right_coord_x,left_coord_x)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(right_coord_x.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,left_coord_x,right_coord_x)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()






cv2.destroyAllWindows()
