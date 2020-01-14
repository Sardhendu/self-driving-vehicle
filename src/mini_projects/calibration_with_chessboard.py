import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8, 3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)



# Make a list of calibration images
images = glob.glob('./data/GO*.jpg')
print(images)


def object_and_image_points(debug=False):
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners 8 corners in the x direction and 6 corners in the y direction
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            if debug:
                cv2.drawChessboardCorners(img, (8,6), corners, ret)
        
    if debug:
        cv2.imshow('img', img)
        cv2.waitKey(500)


