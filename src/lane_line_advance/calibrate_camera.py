
import os
import numpy as np
import cv2
import glob
from src import commons as cm
import itertools


def fetch_object_and_image_points(
        distorted_image_paths, x_corners_cnt, y_corners_cnt, plot=False, dump=True, force_fetch=False
):
    """
    :param distorted_image_paths:
    :param distorted_image_paths:
    :param x_corners_cnt:
    :param y_corners_cnt:
    :param debug:
    :param dump:
    :param force_fetch:
    :return:

    Idea:
        To correct for distortion we need to calibrate the camera and find the calibration co-efficients.
        A way of doing that would be to ge the shift of a pixel coordinate from 1 image to another.
        
        1. We need to identify pixels that are present in all the image (For Chessboard the best pixel marking would
        be the corners). Open CV provides an easy way to find corners in a chessboard
        2. (object_points) We need to give each mark/pint an id. (In this case its the ith_x_value, jth_y_value and 0
        for z axis)
        3. (image_points)We need to find the actual pixel coordinate in the marks/points
    """
    if not force_fetch and os.path.exists(object_image_points_path):
        data_dict = cm.read_pickle(
                save_path=object_image_points_path
        )
        return data_dict["object_points"], data_dict["image_points"]
    object_points = np.zeros((y_corners_cnt * x_corners_cnt, 3), np.float32)
    
    # Get corner coordinates as a mesh grid = (x_coord, y_coord, 0 (xpoint))
    object_points[:, :2] = np.mgrid[0:x_corners_cnt, 0:y_corners_cnt].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all the images.
    object_points_list = []     # 3d points in real world space
    image_points_list = []      # 2d points in image plane.
    
    collect_images_for_plot = []
    for idx, fname in enumerate(distorted_image_paths):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners 8 corners in the x direction and 6 corners in the y direction
        is_valid_corner, image_points = cv2.findChessboardCorners(gray, (x_corners_cnt, y_corners_cnt), None)
        
        # If found, add object points, image points
        if is_valid_corner:
            assert (len(object_points) == len(image_points)), (
                f"len(object_points)={len(object_points)} should equal len(image_points)={len(image_points)}"
            )
            object_points_list.append(object_points)
            image_points_list.append(image_points)

            # Draw and display the corners
            if plot:
                cv2.drawChessboardCorners(img, (x_corners_cnt, y_corners_cnt), image_points, is_valid_corner)
                collect_images_for_plot.append(img)
    if plot:
        fig = cm.subplots(nrows=2, ncols=2, figsize=(10, 10))(collect_images_for_plot[0:4], None)
        cm.save_matplotlib(image_corner_detection_path, fig)
        
    if dump:
        cm.write_pickle(
            save_path=object_image_points_path,
            data_dict={
                "object_points": object_points_list,
                "image_points": image_points_list
            }
        )
    return object_points_list, image_points_list


def calibrate_camera(img_shape, object_points_list, image_points_list, dump=True, force_dump=False):
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
    if not force_dump and os.path.exists(camera_params_path):
        data_dict = cm.read_pickle(
                save_path=camera_params_path
        )
        return data_dict["camera_matrix"], data_dict["distortion_coefficients"]

    ret, camera_matrix, distortion_coefficients, rotational_vector, transient_vector = cv2.calibrateCamera(
            object_points_list, image_points_list, img_shape[1::-1], None, None
    )
    
    if dump:
        cm.write_pickle(
            save_path=camera_params_path,
            data_dict={
                "ret": ret,
                "camera_matrix": camera_matrix,
                "distortion_coefficients": distortion_coefficients,
                "rotational_vector": rotational_vector,
                "transient_vector": transient_vector
            }
        )
    return camera_matrix, distortion_coefficients


def undistort(img, camera_matrix, distortion_coefficients):
    """
    :param img:
    :param camera_matrix:
    :param distortion_coefficients:
    :return:
    """
    undistorted_img = cv2.undistort(img, camera_matrix, distortion_coefficients, None, camera_matrix)
    return undistorted_img


if __name__ == "__main__":
    distorted_image_paths = sorted(glob.glob('./data/camera_cal/calibration*.jpg'))
    print('[Training Data Count]: ', len(distorted_image_paths), distorted_image_paths)
    
    camera_calibration_dir = "./data/camera_calibration_output"
    object_image_points_path = f"{camera_calibration_dir}/object_image_points.pickle"
    camera_params_path = f"{camera_calibration_dir}/camera_params.pickle"
    image_corner_detection_path = f"{camera_calibration_dir}/chessboard_corner_detection.jpg"
    undistorted_img_save_path = f"{camera_calibration_dir}/undistorted_image.jpg"
    
    object_points_list, image_points_list = fetch_object_and_image_points(
            distorted_image_paths, x_corners_cnt=9, y_corners_cnt=6, plot=True, dump=True, force_fetch=True
    )
    camera_matrix, distortion_coefficients = calibrate_camera(
            cm.read_image(distorted_image_paths[0]).shape, object_points_list, image_points_list
    )
    
    undistorted_images = [undistort(
            cm.read_image(img),
            camera_matrix,
            distortion_coefficients)
        for img in distorted_image_paths[0:2]
    ]
    
    fig = cm.subplots(nrows=2, ncols=2, figsize=(10, 10), facecolor='w')(
            list(itertools.chain(*[*zip(
                    [cm.read_image(path) for path in distorted_image_paths[0:2]], undistorted_images
            )]))
    )
    cm.save_matplotlib(undistorted_img_save_path, fig)

