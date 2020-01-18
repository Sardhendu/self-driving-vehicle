import numpy as np
import cv2
import glob
import imageio
from src import commons
import matplotlib.pyplot as plt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

# Make a list of calibration images
image_path = glob.glob('./data/camera_calibration/GO*.jpg')
test_image_path = glob.glob('./data/camera_calibration/test*.jpg')
print('[Training Data Count]: ', len(image_path))
print('[Test Data Count]: ', len(test_image_path))

# read and plot some
# a = [imageio.imread(path_) for num_, path_ in enumerate(image_path) if num_ <= 5]
# commons.subplots(nrows=2, ncols=3, figsize=(30, 30))(a, None)
# plt.show()


def object_and_image_points(debug=False):
    """
    :param debug:
    :return:
    
    Idea:
        In-order to correct for distortion we need to calibrate a camera and find the co-efficients.
        A way of doing that would be to ge the shift of a pixel coordinate from 1 image to another.
        
        1. We need to identify pixels that are present in all the image (For Chessboard the best pixel marking would
        be the corners). Open CV provides an easy way to find corners in a chessboard
        2. (object_points) We need to give each mark/pint an id. (In this case its the ith_x_value, jth_y_value and 0
        for z axis)
        3. (image_points)We need to find the actual pixel coordinate in the marks/points
    """
    num_corners_in_y_axis = 6
    num_corners_in_x_axis = 8
    object_points = np.zeros((num_corners_in_y_axis * num_corners_in_x_axis, 3), np.float32)
    
    # Get corner coordinates as a mesh grid = (x_coord, y_coord, 0 (xpoint))
    object_points[:, :2] = np.mgrid[0:num_corners_in_x_axis, 0:num_corners_in_y_axis].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    object_points_list = [] # 3d points in real world space
    image_points_list = [] # 2d points in image plane.
    collect_images_for_plot = []
    for idx, fname in enumerate(image_path):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners 8 corners in the x direction and 6 corners in the y direction
        is_valid_corner, image_points = cv2.findChessboardCorners(gray, (8, 6), None)
        
        # If found, add object points, image points
        if is_valid_corner:
            assert (len(object_points) == len(image_points))
            object_points_list.append(object_points)
            image_points_list.append(image_points)

            # Draw and display the corners
            if debug:
                cv2.drawChessboardCorners(img, (8,6), image_points, is_valid_corner)
                collect_images_for_plot.append(img)
    if debug:
        commons.subplots(nrows=3, ncols=4, figsize=(50, 30))(collect_images_for_plot[0:12], None)
        plt.show()
    return object_points_list, image_points_list


def calibrate_camera(img_shape, object_points_list, image_points_list):
    """
    :param img_shape:
    :param object_points_list:
    :param image_points_list:
    :return:
    
    Camera matrix =
            | fx  0  cx |    fx and fy are camera focal length
            | 0  fy  cy |    cx x center coordinate of image
            | 0  0   1 |     cy y center coordinate of the image
    Distortion cooeficient:
            [k1, k2, k3, p1, p2]
    rotational_vector: [num_of_images, (r1, r2, r3)]
    transient_vector: [num_of_images, (t1, t2, t3)]
    
    """
    # calibrate_camera_requires image shape in (x, y)
    ret, camera_matrix, distortion_coefficients, rotational_vector, transient_vector = cv2.calibrateCamera(
            object_points_list, image_points_list, img_shape[1::-1], None, None
    )
    return camera_matrix, distortion_coefficients


def undistort_image(img, camera_matrix, distortion_coefficients):
    undistorted_img = cv2.undistort(img, camera_matrix, distortion_coefficients, None, camera_matrix)
    return undistorted_img


img_shape = imageio.imread(image_path[0]).shape
object_points_list, image_points_list = object_and_image_points(debug=False)
camera_matrix, distortion_coefficients = calibrate_camera(img_shape, object_points_list, image_points_list)

test_image = imageio.imread(test_image_path[0])
undistorted_image = undistort_image(test_image, camera_matrix, distortion_coefficients)
print(undistorted_image.shape)
plt.imshow(undistorted_image)
plt.show()

