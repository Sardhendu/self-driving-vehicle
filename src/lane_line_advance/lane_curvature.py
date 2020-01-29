import numpy as np
from src import commons


class ModelParams:
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
    # We are starting the sliding window technique from the lower part of the image. Also Lane Line gradient are
    # most active in the lower part of the image. This is a small hack to give more weight to  pixels at the lower
    # part of the image. We just multiple this vector with the preprocessed warped binary image along y axis.
    y_axis_weights = (np.arange(720)/720).reshape(-1, 1)
    
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
    )).reshape(-1, 1), 720).T * np.tile(np.linspace(1, 7, 720).reshape(-1, 1), 1280)
    hist_weight_matrix /= np.sum(hist_weight_matrix)

    # Smooth curves
    # TODO: Implement moving average
    num_frames = 10
    moving_average_weigths = (np.arange(num_frames)/np.sum(np.arange(num_frames))).reshape(1, -1)
    left_lane_n_polynomial_matrix = np.zeros((720, 10))  # Here 720 is the counts of polynomial points
    right_lane_n_polynomial_matrix = np.zeros((720, 10))
    running_index = 0
    print(moving_average_weigths)
    print(sum(moving_average_weigths))
    # print()
    assert(np.sum(moving_average_weigths) == 1)


def fetch_start_position_with_hist_dist(preprocessed_bin_image, save_dir=None):
    """
    At this point we should have a binary image (1, 0) with detected lane line. Here detections are annotated with 1
    This method is necessary to provide a starting point to find the curvature of the lane line
    :param preprocessed_bin_image:
    :param save_dir:
    :return:
        left_lane_start_point_coordinates = (y1, x1)
        right_lane_start_point_coordinates = (y2, x2)
    # TODO: Lane Line are broad and hence the values between the edges of line lines are 0. It would be a good
    practise to apply a kernel to fill up these valleys
    # TODO: Remove Outliers (Bad Gradients)
    # TODO: How we we handle Bimodal Distribution for a particular lane
    # TODO: Can we try weighted method (Something like a moving Average) with sliding window
    # TODO: Use a kernel to compute a weighted value of neighboring columns instead of using just one.
    """
    # preprocessed_image[preprocessed_image > 0] = 1
    assert(set(np.unique(preprocessed_bin_image)) == {0, 1}), (
        f'The preprocessed image should be binary {0, 1} but contains values {set(np.unique(preprocessed_bin_image))}'
    )
    # Sum all the values in column axis
    frequency_histogram = np.sum(preprocessed_bin_image*ModelParams.hist_weight_matrix, axis=0)
    # Divide the Frequency histogram into two parts to find starting points for Left Lane and Right Lane
    left_lane = frequency_histogram[0: len(frequency_histogram) // 2]
    right_lane = frequency_histogram[len(frequency_histogram) // 2:]
    
    if save_dir:
        fig = commons.graph_subplots(nrows=1, ncols=4, figsize=(50, 10))(
                [frequency_histogram, left_lane, right_lane],
                ["frequency_histogram", "hist_left_lane", "hist_right_lane"]
        )
        fig2 = commons.image_subplots(nrows=1, ncols=1, figsize=(6, 6))(
                [ModelParams.hist_weight_matrix], ["histogram_weight_matrix"]
        )
        commons.save_matplotlib(f"{save_dir}/histogram_dist.png", fig)
        commons.save_matplotlib(f"{save_dir}/histogram_weights.png", fig2)
    
    left_lane_start_index = np.argmax(left_lane)
    right_lane_start_index = len(frequency_histogram) // 2 + np.argmax(right_lane)
    # print(left_lane_start_index, right_lane_start_index)
    return (preprocessed_bin_image.shape[0], left_lane_start_index), (preprocessed_bin_image.shape[0], right_lane_start_index)
    
    
class LaneCurvature:
    def __init__(
            self, preprocessed_bin_image, left_lane_pos_yx, right_lane_pos_yx, window_size, margin, save_dir, pipeline
    ):
        """
        :param preprocessed_bin_image:
        :param left_lane_pos_yx:
        :param right_lane_pos_yx:
        :param window_size:
        :param margin:
        :param save_dir:
        :param pipeline:
        """
        self.preprocessed_bin_image = preprocessed_bin_image
        self.h, self.w = preprocessed_bin_image.shape
        self.left_lane_pos_yx = left_lane_pos_yx
        self.right_lane_pos_yx = right_lane_pos_yx
        self.window_size = window_size
        self.margin = margin
        self.save_dir = save_dir
        self.pipeline = pipeline
        
        if save_dir is not None or self.pipeline != "final":
            self.preprocessed_image_cp = np.dstack((
                self.preprocessed_bin_image, self.preprocessed_bin_image, self.preprocessed_bin_image
            ))
            self.preprocessed_image_cp[self.preprocessed_bin_image == 1] = 255

            self.preprocessed_img_plot = commons.ImagePlots(self.preprocessed_image_cp)
            self.basic_plots = commons.graph_subplots()
    
    def store_curvature_points(self, left_box, right_box):
        # TODO: This process can be initiated by a new thread since this is just storage
        left_box_vals = self.preprocessed_bin_image[left_box[0]:left_box[2], left_box[1]:left_box[3]]
        right_box_vals = self.preprocessed_bin_image[right_box[0]:right_box[2], right_box[1]:right_box[3]]
        
        ll_non_zero_y_idx, ll_non_zero_x_idx = np.where(left_box_vals == 1)
        rl_non_zero_y_idx, rl_non_zero_x_idx = np.where(right_box_vals == 1)

        ModelParams.left_lane_y_points += list(ll_non_zero_y_idx + left_box[0])
        ModelParams.left_lane_x_points += list(ll_non_zero_x_idx + left_box[1])
        ModelParams.right_lane_y_points += list(rl_non_zero_y_idx + right_box[0])
        ModelParams.right_lane_x_points += list(rl_non_zero_x_idx + right_box[1])
        print(f'\n[Curvature Points]\n'
              f'left_lane_y_points = {len(ModelParams.left_lane_y_points)}, '
              f'left_lane_x_points = {len(ModelParams.left_lane_x_points)}, '
              f'right_lane_y_points = {len(ModelParams.right_lane_y_points)} '
              f'right_lane_x_points = {len(ModelParams.right_lane_x_points)}')
        
    def shift_boxes(self, ll_mid_point, rl_mid_point):
        """
        This function adjust the initial rectangular box based on the density of points.
        :param ll_mid_point:
        :param rl_mid_point:
        :return:
        """
        left_box = [
            ll_mid_point[0] - self.window_size[0] // 2, ll_mid_point[1] - self.window_size[1] // 2,
            ll_mid_point[0] + self.window_size[0] // 2, ll_mid_point[1] + self.window_size[1] // 2,
        ]
        right_box = [
            rl_mid_point[0] - self.window_size[0] // 2, rl_mid_point[1] - self.window_size[1] // 2,
            rl_mid_point[0] + self.window_size[0] // 2, rl_mid_point[1] + self.window_size[1] // 2,
        ]

        return left_box, right_box
        
    def find_lane_points_with_sliding_window(self):
        """
        Finds lane points using sliding window technique
        :return:
        """
        # TODO: When there are no activated pixels for a box, then it is a good idea to to take weighted average of
        #  the last 2-3 boxes
        ll_y_mid_point = self.left_lane_pos_yx[0] - (self.window_size[0]//2)
        ll_x_mid_point = self.left_lane_pos_yx[1]
    
        rl_y_mid_point = self.right_lane_pos_yx[0] - (self.window_size[0] // 2)
        rl_x_mid_point = self.right_lane_pos_yx[1]
        
        cnt = 0
        while cnt <= 9:
            left_box = [
                ll_y_mid_point - self.window_size[0] // 2, ll_x_mid_point - self.window_size[1] // 2,
                ll_y_mid_point + self.window_size[0] // 2, ll_x_mid_point + self.window_size[1] // 2,
            ]
            right_box = [
                rl_y_mid_point - self.window_size[0] // 2, rl_x_mid_point - self.window_size[1] // 2,
                rl_y_mid_point + self.window_size[0] // 2, rl_x_mid_point + self.window_size[1] // 2,
            ]

            left_box_vals = self.preprocessed_bin_image[left_box[0]:left_box[2], left_box[1]:left_box[3]]
            right_box_vals = self.preprocessed_bin_image[right_box[0]:right_box[2], right_box[1]:right_box[3]]
            
            _, ll_non_zero_x_idx = np.where(left_box_vals == 1)
            _, rl_non_zero_x_idx = np.where(right_box_vals == 1)

            # Center pxl point of all the x that have activated pixels
            if len(ll_non_zero_x_idx) > 0:
                ll_x_mid_point = np.int(np.mean(ll_non_zero_x_idx)) + left_box[1]
            if len(rl_non_zero_x_idx) > 0:
                rl_x_mid_point = np.int(np.mean(rl_non_zero_x_idx)) + right_box[1]
            left_box, right_box = self.shift_boxes(
                    ll_mid_point=[ll_y_mid_point, ll_x_mid_point], rl_mid_point=[rl_y_mid_point, rl_x_mid_point]
            )
            self.store_curvature_points(left_box, right_box)
            
            ll_y_mid_point = ll_y_mid_point - self.window_size[0]
            rl_y_mid_point = rl_y_mid_point - self.window_size[0]
            
            if self.save_dir or self.pipeline != "final":
                self.preprocessed_img_plot.rectangle(left_box)
                self.preprocessed_img_plot.rectangle(right_box)
                
            cnt += 1
            
    def find_lane_points_with_prior_lane_points(self):
        """
        Lanes don't change abruptly, hence it is a good idea to use the previous detected lane lines with an extended
        margin as our new search space to find new polynomial.
        :return:
        """
        activated_pixel_idx = self.preprocessed_bin_image.nonzero()
        y_active_idx = np.array(activated_pixel_idx[0])
        x_active_idx = np.array(activated_pixel_idx[1])

        # The below may seem like a redundant step but this is required because we need an array thats the same size as
        # x_active_idx and y_active_idx because finding if arr_1 points are smaller/larger that arr_2 is expensive
        # operation when len(arr_1) != len(arr_2). In order to achive this using mask we may have to create a mesh grip
        # Seeing all option this option seems faster
        x_left_pred_idx = ModelParams.left_fit_pxl_curr[0] * y_active_idx ** 2 + ModelParams.left_fit_pxl_curr[1] * y_active_idx + ModelParams.left_fit_pxl_curr[2]
        x_right_pred_idx = ModelParams.right_fit_pxl_curr[0] * y_active_idx ** 2 + ModelParams.right_fit_pxl_curr[1] * y_active_idx + ModelParams.right_fit_pxl_curr[2]
        
        ll_inds = (
                (x_active_idx > x_left_pred_idx - self.margin) & (x_active_idx < x_left_pred_idx + self.margin)
        )
        rr_inds = (
                (x_active_idx > x_right_pred_idx - self.margin) & (x_active_idx < x_right_pred_idx + self.margin)
        )
        
        ModelParams.left_lane_y_points = y_active_idx[ll_inds]
        ModelParams.left_lane_x_points = x_active_idx[ll_inds]
        ModelParams.right_lane_y_points = y_active_idx[rr_inds]
        ModelParams.right_lane_x_points = x_active_idx[rr_inds]

    def find_lane_points(self):
        if (
                len(ModelParams.left_lane_y_points) == 0 or
                len(ModelParams.left_lane_x_points) == 0 or
                len(ModelParams.right_lane_y_points) == 0 or
                len(ModelParams.right_lane_x_points) == 0
        ):
            self.find_lane_points_with_sliding_window()
        else:
            self.find_lane_points_with_prior_lane_points()
            
    def fit(self, degree=2):
        ModelParams.left_fit_pxl_curr = np.polyfit(ModelParams.left_lane_y_points, ModelParams.left_lane_x_points, deg=degree)
        ModelParams.right_fit_pxl_curr = np.polyfit(ModelParams.right_lane_y_points, ModelParams.right_lane_x_points, deg=degree)

        ModelParams.left_fit_pxl_coefficients.append(ModelParams.left_fit_pxl_curr)
        ModelParams.right_fit_pxl_coefficients.append(ModelParams.right_fit_pxl_curr)
        assert (len(ModelParams.left_lane_y_points) == len(ModelParams.left_lane_x_points))
        assert (len(ModelParams.right_lane_y_points) == len(ModelParams.right_lane_x_points))
        
    def predict(self):
        """
        y = b0 + b1*x, b2*x_sq
        :return:
        """
        y_new = np.linspace(0, self.preprocessed_bin_image.shape[0] - 1, self.preprocessed_bin_image.shape[0])
        try:
            left_x_new = ModelParams.left_fit_pxl_curr[0] * y_new ** 2 + ModelParams.left_fit_pxl_curr[1] * y_new + ModelParams.left_fit_pxl_curr[2]
            right_x_new = ModelParams.right_fit_pxl_curr[0] * y_new ** 2 + ModelParams.right_fit_pxl_curr[1] * y_new + ModelParams.right_fit_pxl_curr[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_x_new = 1 * y_new ** 2 + 1 * y_new
            right_x_new = 1 * y_new ** 2 + 1 * y_new

        # Overwrite the predicted points as new search space
        ModelParams.left_lane_x_points = left_x_new
        ModelParams.left_lane_y_points = y_new
        ModelParams.right_lane_x_points = right_x_new
        ModelParams.right_lane_y_points = y_new
        
        assert (len(ModelParams.left_lane_y_points) == len(ModelParams.right_lane_y_points) == len(
                ModelParams.left_lane_x_points) == len(ModelParams.left_lane_y_points))

        left_x_new, right_x_new = self.average_n_lanes(left_x_new, right_x_new)
        return y_new, left_x_new, right_x_new
    
    def average_n_lanes(self, left_x_new, right_x_new):
        # print('RUNNING ===========> ', ModelParams.running_index)
        # print(np.sum(left_x_new))

        ModelParams.left_lane_n_polynomial_matrix[:, :-1] = ModelParams.left_lane_n_polynomial_matrix[:, 1:]
        ModelParams.left_lane_n_polynomial_matrix[:, -1:] = left_x_new.reshape(-1, 1)

        ModelParams.right_lane_n_polynomial_matrix[:, :-1] = ModelParams.right_lane_n_polynomial_matrix[:, 1:]
        ModelParams.right_lane_n_polynomial_matrix[:, -1:] = right_x_new.reshape(-1, 1)
        
        # print(np.sum(ModelParams.left_lane_n_polynomial_matrix, axis=0))
        if ModelParams.running_index >= 9:
            # print( ModelParams.left_lane_n_polynomial_matrix.shape, ModelParams.moving_average_weigths.shape)
            
            left_x_new = np.sum(ModelParams.left_lane_n_polynomial_matrix * ModelParams.moving_average_weigths, axis=1)
            right_x_new = np.sum(ModelParams.right_lane_n_polynomial_matrix * ModelParams.moving_average_weigths,
                                 axis=1)
            # print(left_x_new.shape)
            # print('ModelParams.left_lane_n_polynomial_matrix: \n', ModelParams.left_lane_n_polynomial_matrix)

        ModelParams.running_index += 1

        return left_x_new, right_x_new
    
    def measure_radius_in_pxl(self):
        """
        This function computes the Radius of LaneCurvature in pixels
        :return:
        """
        lb2, lb1, _ = ModelParams.left_fit_pxl_curr
        rb2, rb1, _ = ModelParams.right_fit_pxl_curr
        ModelParams.left_lane_curvature_radii_curr = ((1 + (2 * lb2 * self.h + lb1) ** 2) ** 1.5) / np.absolute(
            2 * lb2)
        ModelParams.right_lane_curvature_radii_curr = ((1 + (2 * rb2 * self.h + rb1) ** 2) ** 1.5) / np.absolute(
            2 * rb2)
        # Take any y value, here we take the max
        ModelParams.left_lane_curvature_radii.append(ModelParams.left_lane_curvature_radii_curr)
        ModelParams.right_lane_curvature_radii.append(ModelParams.right_lane_curvature_radii_curr)
            
    def measure_radius_in_meter(self):
        """
        Here we project the y_points and x_points of both the lanes from the warped pxl coordinate system to the real
        world coordinate space with distance in pxl. Then we fit a polynomial and compute the radius of curvature.
        
        It would not be wise to use the polyfit used on warped image pxl space and convert that to meters. Because
        small turns in warped image can imply sharper turns in the real world.
        :return:
        """
        ModelParams.left_fit_meter_curr = np.polyfit(
                ModelParams.left_lane_y_points*ModelParams.ym_per_pix,
                ModelParams.left_lane_x_points*ModelParams.xm_per_pix,
                deg=2
        )
        
        ModelParams.right_fit_meter_curr = np.polyfit(
                ModelParams.right_lane_y_points*ModelParams.ym_per_pix,
                ModelParams.right_lane_x_points*ModelParams.xm_per_pix,
                deg=2
        )
        lb2, lb1, _ = ModelParams.left_fit_meter_curr
        rb2, rb1, _ = ModelParams.right_fit_meter_curr

        ModelParams.left_lane_curvature_radii_curr = ((1 + (2 * lb2 * self.h * ModelParams.ym_per_pix + lb1) ** 2) ** 1.5) / np.absolute(2 * lb2)
        ModelParams.right_lane_curvature_radii_curr = ((1 + (2 * rb2 * self.h * ModelParams.ym_per_pix + rb1) ** 2) ** 1.5) / np.absolute(2 * rb2)
        # Take any y value, here we take the max
        ModelParams.left_lane_curvature_radii.append(ModelParams.left_lane_curvature_radii_curr)
        ModelParams.right_lane_curvature_radii.append(ModelParams.right_lane_curvature_radii_curr)
    
    def plot(self, left_y_new, left_x_new, right_y_new, right_x_new):
        draw_points = np.asarray([left_x_new, left_y_new]).T.astype(np.int32)
        self.preprocessed_img_plot.polylines(draw_points, (50, 255, 255))
        draw_points = np.asarray([right_x_new, right_y_new]).T.astype(np.int32)
        self.preprocessed_img_plot.polylines(draw_points, (50, 255, 255))

        if self.save_dir:
            commons.save_image(f"{self.save_dir}/curvature_windows.png", self.preprocessed_img_plot.image)

        

# from src import commons

#
# test_image_name = "test4"#"straight_lines1"  # test4
#
# input_image_path = f'./data/test_images/{test_image_name}.jpg'
# output_plot_path = f'./data/output_images/{test_image_name}.png'
# output_plot_path2 = f'./data/output_images/postprocess_{test_image_name}.png'
# hist_output_path = f"./data/output_images/hist_{test_image_name}.png"
# curvature_bbox_output_path = f"./data/output_images/curvature_{test_image_name}.png"
#
# image = commons.read_image(input_image_path)
# preprocess_pipeline = PreprocessingPipeline(image, save_path=output_plot_path)
# postprocess_pipeline = PostprocessingPipeline(image, save_path=output_plot_path2)
#
# preprocess_pipeline.warp()
# preprocessed_bin_image = preprocess_pipeline.preprocess()
# preprocessed_bin_image = preprocessed_bin_image.astype(np.int32)
# print('preprocessed_img: ', np.unique(preprocessed_bin_image))
#
# left_lane_pos_yx, right_lane_pos_yx = fetch_start_position_with_hist_dist(
#         preprocessed_bin_image.copy(), save_path=hist_output_path
# )
#
# obj_l_curv = LaneCurvature(
#     preprocessed_bin_image=preprocessed_bin_image,
#     left_lane_pos_yx=left_lane_pos_yx,
#     right_lane_pos_yx=right_lane_pos_yx,
#     window_size=(70, 130),
#     save_path=curvature_bbox_output_path
# )
# obj_l_curv.find_lane_points()
# obj_l_curv.fit()
# y_new, left_x_new, right_x_new = obj_l_curv.predict()
# obj_l_curv.plot(y_new, left_x_new, y_new, right_x_new)
# postprocess_pipeline.unwarp()
# postprocess_pipeline.transform_lane_points(
#         left_lane_points=np.column_stack((left_x_new, y_new)), right_lane_points=np.column_stack((right_x_new, y_new))
# )
# postprocess_pipeline.plot()





    