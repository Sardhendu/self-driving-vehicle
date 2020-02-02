import numpy as np
import cv2


class CameraParams:
    camera_matrix = np.array([
        [1.15777930e+03, 0.00000000e+00, 6.67111054e+02],
        [0.00000000e+00, 1.15282291e+03, 3.86128937e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ], dtype=float)
    distortion_coefficients = np.array([
        [-0.24688775, - 0.02373134, - 0.00109842,  0.00035108, - 0.00258569]
    ], dtype=float)


class PipelineParams:
    height = 720
    width = 1280
    src_points = [(500, 450), (0, 700), (780, 450), (1280, 700)]
    # self.src_points = [(500, 450), (0, 700), (750, 450), (1250, 700)]
    dst_points = [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]
    
    red_threshold = [(150, 255), (130, 255)]
    saturation_threshold = [(100, 255), (50, 255)]
    gradient_threshold = [(15, 255), (10, 255)]
    M = cv2.getPerspectiveTransform(
            np.array(src_points).astype(np.float32), np.array(dst_points).astype(np.float32)
    )
    
    M_inv = cv2.getPerspectiveTransform(
            np.array(dst_points).astype(np.float32), np.array(src_points).astype(np.float32)
    )


class CurvatureParams:
    window_size_yx = (70, 130)
    search_loops = np.floor(720 / window_size_yx[0])
    left_lane_y_points = []
    left_lane_x_points = []
    right_lane_y_points = []
    right_lane_x_points = []
    
    left_lane_curvature_radii = []
    right_lane_curvature_radii = []
    left_lane_curvature_radii_curr = None
    right_lane_curvature_radii_curr = None
    
    left_fit_pxl_curr = None
    right_fit_pxl_curr = None
    left_fit_meter_curr = None
    right_fit_meter_curr = None
    
    # Capture all the b0, b1 and b2 for left and right lane poly fit
    left_fit_pxl_coefficients = []
    right_fit_pxl_coefficients = []
    
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    
    # -------------------------------------------------------------------------------------------
    # Some Awesome Hacks
    # -------------------------------------------------------------------------------------------
    # Assumption: The camera is centered on the car. Lane lines edges in the warped image are broad mostly at the lower
    # region and they narrow as they move up in the image. Here we define a kernel of certain size. The idea is
    # instead of sum each y column of the image we make
    # the weighted sum inside the kernel
    hist_weight_matrix = np.tile(np.concatenate((
        np.linspace(1, 2, 160),
        np.linspace(2, 3, 160),
        np.linspace(3, 2, 160),
        np.linspace(2, 1, 160),
        np.linspace(1, 2, 160),
        np.linspace(2, 3, 160),
        np.linspace(3, 2, 160),
        np.linspace(2, 1, 160)
    )).reshape(-1, 1), 720).T * np.tile(np.linspace(1, 5, 720).reshape(-1, 1), 1280)
    hist_weight_matrix /= np.sum(hist_weight_matrix)
    
    # Smooth curves
    # TODO: Implement moving average
    num_frames = 10
    frame_weights = np.array([2, 2, 3, 3, 4, 4, 5, 5, 6, 7], dtype=float)
    moving_average_weigths = (np.array(frame_weights) / np.sum(frame_weights)).reshape(1, -1)
    left_lane_n_polynomial_matrix = np.zeros((720, 10))  # Here 720 is the counts of polynomial points
    right_lane_n_polynomial_matrix = np.zeros((720, 10))
    running_index = 0
    assert (num_frames == len(frame_weights))
    print(np.sum(moving_average_weigths))
    assert (np.sum(moving_average_weigths) >= 0.9)
    
    # Sanity Check Parameters
    ll_margin_with_no_points = 0
    rl_margin_with_no_points = 0
    
    # Capture the variance of the current line at (t) wrt to previous line (t-1)
    # Since line at (t-1) is actually the weighted average of the previous (t-9)
    # we are actually calculating a weighted variance
    left_line_curr_poly_variance = []
    right_line_curr_poly_variance = []
    max_variance = 0.1
    