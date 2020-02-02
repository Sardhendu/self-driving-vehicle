import numpy as np
from src import commons
from src.lane_line_advance.params import CurvatureParams, PipelineParams


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
    frequency_histogram = np.sum(preprocessed_bin_image*CurvatureParams.hist_weight_matrix, axis=0)
    # Divide the Frequency histogram into two parts to find starting points for Left Lane and Right Lane
    left_lane = frequency_histogram[0: len(frequency_histogram) // 2]
    right_lane = frequency_histogram[len(frequency_histogram) // 2:]
    
    if save_dir:
        fig = commons.graph_subplots(nrows=1, ncols=3, figsize=(50, 10))(
                [frequency_histogram, left_lane, right_lane],
                ["frequency_histogram", "hist_left_lane", "hist_right_lane"]
        )
        fig2 = commons.image_subplots(nrows=1, ncols=1, figsize=(6, 6))(
                [CurvatureParams.hist_weight_matrix], ["histogram_weight_matrix"]
        )
        commons.save_matplotlib(f"{save_dir}/histogram_dist.png", fig)
        commons.save_matplotlib(f"{save_dir}/histogram_weights.png", fig2)
    
    left_lane_start_index = np.argmax(left_lane)
    right_lane_start_index = len(frequency_histogram) // 2 + np.argmax(right_lane)
    # print('Left lane right lane start: ', left_lane_start_index, right_lane_start_index)
    return (preprocessed_bin_image.shape[0], left_lane_start_index), (preprocessed_bin_image.shape[0], right_lane_start_index)
    
    
class LaneCurvature:
    def __init__(
            self, preprocessed_bin_image, left_lane_pos_yx, right_lane_pos_yx, margin, save_dir, pipeline
    ):
        """
        :param preprocessed_bin_image:
        :param left_lane_pos_yx:
        :param right_lane_pos_yx:
        :param margin:
        :param save_dir:
        :param pipeline:
        """
        self.preprocessed_bin_image = preprocessed_bin_image
        self.h, self.w = preprocessed_bin_image.shape
        self.left_lane_pos_yx = left_lane_pos_yx
        self.right_lane_pos_yx = right_lane_pos_yx
        self.margin = margin
        self.save_dir = save_dir
        self.pipeline = pipeline
        self.left_xy_margin_left = []
        self.left_xy_margin_right = []
        self.right_xy_margin_left = []
        self.right_xy_margin_right = []

        if save_dir is not None or self.pipeline != "final":
            self.preprocessed_image_cp = np.dstack((
                self.preprocessed_bin_image, self.preprocessed_bin_image, self.preprocessed_bin_image
            ))
            self.preprocessed_image_cp[self.preprocessed_bin_image == 1] = 255

            self.preprocessed_img_plot = commons.ImagePlots(self.preprocessed_image_cp)
            self.basic_plots = commons.graph_subplots()

            self.dummy_plot = np.zeros(self.preprocessed_image_cp.shape)
    
    def shift_boxes(self, ll_mid_point, rl_mid_point):
        """
        This function adjust the initial rectangular box based on the density of points.
        :param ll_mid_point:
        :param rl_mid_point:
        :return:
        """
        left_box = [
            ll_mid_point[0] - CurvatureParams.window_size_yx[0] // 2, ll_mid_point[1] - CurvatureParams.window_size_yx[1] // 2,
            ll_mid_point[0] + CurvatureParams.window_size_yx[0] // 2, ll_mid_point[1] + CurvatureParams.window_size_yx[1] // 2,
        ]
        right_box = [
            rl_mid_point[0] - CurvatureParams.window_size_yx[0] // 2, rl_mid_point[1] - CurvatureParams.window_size_yx[1] // 2,
            rl_mid_point[0] + CurvatureParams.window_size_yx[0] // 2, rl_mid_point[1] + CurvatureParams.window_size_yx[1] // 2,
        ]

        return left_box, right_box
        
    def find_lane_points_with_sliding_window(self, which_lane, mode):
        """
        Finds lane points using sliding window technique
        :return:
        """
        # TODO: When there are no activated pixels for a box, then it is a good idea to to take weighted average of
        #  the last 2-3 boxes
        ll_y_mid_point = self.left_lane_pos_yx[0] - (CurvatureParams.window_size_yx[0] // 2)
        ll_x_mid_point = self.left_lane_pos_yx[1]
    
        rl_y_mid_point = self.right_lane_pos_yx[0] - (CurvatureParams.window_size_yx[0] // 2)
        rl_x_mid_point = self.right_lane_pos_yx[1]
        cnt = 0

        left_box = [
            ll_y_mid_point - CurvatureParams.window_size_yx[0] // 2, ll_x_mid_point - CurvatureParams.window_size_yx[1] // 2,
            ll_y_mid_point + CurvatureParams.window_size_yx[0] // 2, ll_x_mid_point + CurvatureParams.window_size_yx[1] // 2,
        ]
        right_box = [
            rl_y_mid_point - CurvatureParams.window_size_yx[0] // 2, rl_x_mid_point - CurvatureParams.window_size_yx[1] // 2,
            rl_y_mid_point + CurvatureParams.window_size_yx[0] // 2, rl_x_mid_point + CurvatureParams.window_size_yx[1] // 2,
        ]

        if self.save_dir:
            self.preprocessed_img_plot.rectangle(left_box, color=(0, 255, 0))
            self.preprocessed_img_plot.rectangle(right_box, color=(0, 255, 0))

        left_lane_y_points = []
        left_lane_x_points = []
        right_lane_y_points = []
        right_lane_x_points = []
        while cnt <= CurvatureParams.search_loops:
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

            # --------------------------------------------------------------------------------------------
            # Get points inside the Rectangle Bboxes
            # --------------------------------------------------------------------------------------------
            left_box_vals = self.preprocessed_bin_image[left_box[0]:left_box[2], left_box[1]:left_box[3]]
            right_box_vals = self.preprocessed_bin_image[right_box[0]:right_box[2], right_box[1]:right_box[3]]

            ll_non_zero_y_idx, ll_non_zero_x_idx = np.where(left_box_vals == 1)
            rl_non_zero_y_idx, rl_non_zero_x_idx = np.where(right_box_vals == 1)

            left_lane_y_points += list(ll_non_zero_y_idx + left_box[0])
            left_lane_x_points += list(ll_non_zero_x_idx + left_box[1])
            right_lane_y_points += list(rl_non_zero_y_idx + right_box[0])
            right_lane_x_points += list(rl_non_zero_x_idx + right_box[1])
            
            if mode=="debug":
                print(f'\n[Curvature Points]\n'
                      f'left_lane_y_points = {len(left_lane_y_points)}, '
                      f'left_lane_x_points = {len(left_lane_x_points)}, '
                      f'right_lane_y_points = {len(right_lane_y_points)} '
                      f'right_lane_x_points = {len(right_lane_x_points)}')
            
            if self.save_dir or self.pipeline != "final":
                self.preprocessed_img_plot.rectangle(left_box, color=(255, 0, 0))
                self.preprocessed_img_plot.rectangle(right_box, color=(255, 0, 0))
                self.preprocessed_img_plot.point([ll_x_mid_point, ll_y_mid_point])
                self.preprocessed_img_plot.point([rl_x_mid_point, rl_y_mid_point])
                
            ll_y_mid_point = ll_y_mid_point - CurvatureParams.window_size_yx[0]
            rl_y_mid_point = rl_y_mid_point - CurvatureParams.window_size_yx[0]
            cnt += 1

        if which_lane == "left" or which_lane == "both":
            CurvatureTools.set_curr_left_lane_points(left_x_new=left_lane_x_points, left_y_new=left_lane_y_points)
            
        if which_lane == "right" or which_lane == "both":
            CurvatureTools.set_curr_right_lane_points(right_x_new=right_lane_x_points, right_y_new=right_lane_y_points)

    def find_lane_points_with_prior_lane_points(self, mode):
        """
        Lanes don't change abruptly, hence it is a good idea to use the previous detected lane lines with an extended
        margin as our new search space to find new polynomial. This function uses the previous predicted lane line,
        creates a margin around it, and extract all the active pixels falling inside the margin for both left and right
        lanes.
        :return:
        """
        # print('left_lane_y_points: \n', len(CurvatureParams.left_lane_y_points),
        #       len(CurvatureParams.right_lane_y_points))
        # print(1 / 0)
        activated_pixel_idx = self.preprocessed_bin_image.nonzero()
        y_active_idx = np.array(activated_pixel_idx[0])
        x_active_idx = np.array(activated_pixel_idx[1])

        left_x_points_m1 = CurvatureParams.left_lane_x_points - self.margin
        left_x_points_m2 = CurvatureParams.left_lane_x_points + self.margin
        right_x_points_m1 = CurvatureParams.right_lane_x_points - self.margin
        right_x_points_m2 = CurvatureParams.right_lane_x_points + self.margin

        left_x_points_m1_ext = left_x_points_m1[y_active_idx]
        left_x_points_m2_ext = left_x_points_m2[y_active_idx]
        right_x_points_m1_ext = right_x_points_m1[y_active_idx]
        right_x_points_m2_ext = right_x_points_m2[y_active_idx]

        ll_inds = (
                (x_active_idx > left_x_points_m1_ext) & (x_active_idx < left_x_points_m2_ext)
        )

        rl_inds = (
                (x_active_idx > right_x_points_m1_ext) & (x_active_idx < right_x_points_m2_ext)
        )
        
        if mode == "warped":
            self.left_xy_margin_left = np.column_stack((left_x_points_m1, CurvatureParams.right_lane_y_points)).astype(np.int32)
            self.left_xy_margin_right = np.column_stack((left_x_points_m2, CurvatureParams.right_lane_y_points)).astype(np.int32)
            self.right_xy_margin_left = np.column_stack((right_x_points_m1, CurvatureParams.right_lane_y_points)).astype(
                    np.int32)
            self.right_xy_margin_right = np.column_stack((right_x_points_m2, CurvatureParams.right_lane_y_points)).astype(
                    np.int32)
    
        if sum(ll_inds) > 0:
            CurvatureTools.set_curr_left_lane_points(left_x_new=x_active_idx[ll_inds], left_y_new=y_active_idx[ll_inds])
        else:
            print('WARNING WARNING ==================> NO (Left Lane XY-points) FOUND')
    
        if sum(rl_inds) > 0:
            CurvatureTools.set_curr_right_lane_points(right_x_new=x_active_idx[rl_inds],
                                                      right_y_new=y_active_idx[rl_inds])
        else:
            print('WARNING WARNING ==================> NO (Right Lane XY-points) FOUND')

    def find_lane_points(self, mode):
        if (
                # len(CurvatureParams.left_lane_y_points) == 0 and
                len(CurvatureParams.left_lane_x_points) == 0 and
                # len(CurvatureParams.right_lane_y_points) == 0 and
                len(CurvatureParams.right_lane_x_points) == 0
        ):
            self.find_lane_points_with_sliding_window(which_lane="both", mode=mode)
        elif len(CurvatureParams.left_lane_x_points) == 0:
            self.find_lane_points_with_sliding_window(which_lane="left", mode=mode)
        elif len(CurvatureParams.right_lane_x_points) == 0:
            self.find_lane_points_with_sliding_window(which_lane="right", mode=mode)
        else:
            self.find_lane_points_with_prior_lane_points(mode)
            
    def fit(self, degree=2):
        CurvatureParams.left_fit_pxl_curr = np.polyfit(CurvatureParams.left_lane_y_points, CurvatureParams.left_lane_x_points, deg=degree)
        CurvatureParams.right_fit_pxl_curr = np.polyfit(CurvatureParams.right_lane_y_points, CurvatureParams.right_lane_x_points, deg=degree)

        CurvatureParams.left_fit_pxl_coefficients.append(CurvatureParams.left_fit_pxl_curr)
        CurvatureParams.right_fit_pxl_coefficients.append(CurvatureParams.right_fit_pxl_curr)
        assert (len(CurvatureParams.left_lane_y_points) == len(CurvatureParams.left_lane_x_points))
        assert (len(CurvatureParams.right_lane_y_points) == len(CurvatureParams.right_lane_x_points))
        
    def predict(self):
        """
        y = b0 + b1*x, b2*x_sq
        :return:
        """
        y_new = np.linspace(0, self.preprocessed_bin_image.shape[0] - 1, self.preprocessed_bin_image.shape[0])
        try:
            left_x_new = CurvatureParams.left_fit_pxl_curr[0] * y_new ** 2 + CurvatureParams.left_fit_pxl_curr[1] * y_new + CurvatureParams.left_fit_pxl_curr[2]
            right_x_new = CurvatureParams.right_fit_pxl_curr[0] * y_new ** 2 + CurvatureParams.right_fit_pxl_curr[1] * y_new + CurvatureParams.right_fit_pxl_curr[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_x_new = 1 * y_new ** 2 + 1 * y_new
            right_x_new = 1 * y_new ** 2 + 1 * y_new

        # Sometimes the predictions can go way off that the actual values
        left_x_new, right_x_new = CurvatureTools.bound_predictions(left_x_new, right_x_new)

        # ---------------------------------------------------------------------------------------------------------
        # Check lines with variance of curvature
        # ---------------------------------------------------------------------------------------------------------
        # print('right_x_new: \n', right_x_new)
        left_x_new, right_x_new = CurvatureTools.check_lines_with_variance_of_curvature_change(left_x_new, right_x_new)
        CurvatureTools.set_curr_left_lane_points(left_x_new=left_x_new, left_y_new=y_new)
        CurvatureTools.set_curr_right_lane_points(right_x_new=right_x_new, right_y_new=y_new)

        # ---------------------------------------------------------------------------------------------------------
        # Average N-Lane lines
        # ---------------------------------------------------------------------------------------------------------
        """
        Step 1: We shift the polynomial matrix [:, 1:n] to [:, 0:n-1]
        Step 2: We set the new left, right lane points at [:, nth] y position
        Step 3: We compute the nth weighted average for both left and right Lane lines
        Step 4: We reset the left, right lane points at [:, nth] y position with the new averages lane lines
        """
        CurvatureTools.shift_polynomial_matrix()
        CurvatureTools.set_curr_lane_points_on_polynomial_matrix(left_x_new=left_x_new, right_x_new=right_x_new)
        left_x_new, right_x_new = CurvatureTools.average_n_lanes(left_x_new, right_x_new)
        CurvatureTools.set_curr_lane_points_on_polynomial_matrix(left_x_new=left_x_new, right_x_new=right_x_new)
        CurvatureTools.set_curr_left_lane_points(left_x_new=left_x_new, left_y_new=y_new)
        CurvatureTools.set_curr_right_lane_points(right_x_new=right_x_new, right_y_new=y_new)
        
        assert (len(CurvatureParams.left_lane_y_points) == len(CurvatureParams.right_lane_y_points) == len(
                CurvatureParams.left_lane_x_points) == len(CurvatureParams.left_lane_y_points)), (
            f'len(CurvatureParams.left_lane_y_points)={len(CurvatureParams.left_lane_y_points)}, '
            f'len(CurvatureParams.right_lane_y_points)={len(CurvatureParams.right_lane_y_points)}, '
            f'len(CurvatureParams.left_lane_x_points)={len(CurvatureParams.left_lane_x_points)}, '
            f'len(CurvatureParams.left_lane_y_points)={len(CurvatureParams.left_lane_y_points)}'
        )

        CurvatureParams.running_index += 1
        return y_new, left_x_new, right_x_new
    
    def measure_radius_in_pxl(self):
        """
        This function computes the Radius of LaneCurvature in pixels
        :return:
        """
        lb2, lb1, _ = CurvatureParams.left_fit_pxl_curr
        rb2, rb1, _ = CurvatureParams.right_fit_pxl_curr
        CurvatureParams.left_lane_curvature_radii_curr = ((1 + (2 * lb2 * self.h + lb1) ** 2) ** 1.5) / np.absolute(
            2 * lb2)
        CurvatureParams.right_lane_curvature_radii_curr = ((1 + (2 * rb2 * self.h + rb1) ** 2) ** 1.5) / np.absolute(
            2 * rb2)
        # Take any y value, here we take the max
        CurvatureParams.left_lane_curvature_radii.append(CurvatureParams.left_lane_curvature_radii_curr)
        CurvatureParams.right_lane_curvature_radii.append(CurvatureParams.right_lane_curvature_radii_curr)
        
    def measure_radius_in_meter(self):
        """
        Here we project the y_points and x_points of both the lanes from the warped pxl coordinate system to the real
        world coordinate space with distance in pxl. Then we fit a polynomial and compute the radius of curvature.
        
        It would not be wise to use the polyfit used on warped image pxl space and convert that to meters. Because
        small turns in warped image can imply sharper turns in the real world.
        :return:
        """
        CurvatureParams.left_fit_meter_curr = np.polyfit(
                CurvatureParams.left_lane_y_points*CurvatureParams.ym_per_pix,
                CurvatureParams.left_lane_x_points*CurvatureParams.xm_per_pix,
                deg=2
        )
        
        CurvatureParams.right_fit_meter_curr = np.polyfit(
                CurvatureParams.right_lane_y_points*CurvatureParams.ym_per_pix,
                CurvatureParams.right_lane_x_points*CurvatureParams.xm_per_pix,
                deg=2
        )
        lb2, lb1, _ = CurvatureParams.left_fit_meter_curr
        rb2, rb1, _ = CurvatureParams.right_fit_meter_curr

        CurvatureParams.left_lane_curvature_radii_curr = ((1 + (2 * lb2 * self.h * CurvatureParams.ym_per_pix + lb1) ** 2) ** 1.5) / np.absolute(2 * lb2)
        CurvatureParams.right_lane_curvature_radii_curr = ((1 + (2 * rb2 * self.h * CurvatureParams.ym_per_pix + rb1) ** 2) ** 1.5) / np.absolute(2 * rb2)
        # Take any y value, here we take the max
        CurvatureParams.left_lane_curvature_radii.append(CurvatureParams.left_lane_curvature_radii_curr)
        CurvatureParams.right_lane_curvature_radii.append(CurvatureParams.right_lane_curvature_radii_curr)
        
    def plot(self, left_y_new, left_x_new, right_y_new, right_x_new):
        draw_points = np.asarray([left_x_new, left_y_new]).T.astype(np.int32)
        self.preprocessed_img_plot.polylines(draw_points, (50, 255, 255))
        draw_points = np.asarray([right_x_new, right_y_new]).T.astype(np.int32)
        self.preprocessed_img_plot.polylines(draw_points, (50, 255, 255))

        if len(self.left_xy_margin_left) > 0:
            self.preprocessed_img_plot.polylines(self.left_xy_margin_left, (255, 0, 0), thickness=1)
            self.preprocessed_img_plot.polylines(self.left_xy_margin_right, (255, 0, 0), thickness=1)
            
        if len(self.right_xy_margin_left) > 0:
            self.preprocessed_img_plot.polylines(self.right_xy_margin_left, (255, 0, 0), thickness=1)
            self.preprocessed_img_plot.polylines(self.right_xy_margin_right, (255, 0, 0), thickness=1)

        if self.save_dir:
            commons.save_image(f"{self.save_dir}/curvature_windows.png", self.preprocessed_img_plot.image)


class CurvatureTools:
    @staticmethod
    def bound_predictions(left_x_new, right_x_new):
        return (
            np.minimum(
                    np.maximum(
                            left_x_new,
                            np.tile(PipelineParams.src_points[1][0], len(left_x_new))),
                    PipelineParams.src_points[3][0]
            ),
            np.maximum(
                    np.minimum(
                            right_x_new,
                            np.tile(PipelineParams.src_points[3][0], len(right_x_new))),
                    PipelineParams.src_points[1][0]
            )
        )
    
    @staticmethod
    def average_n_lanes(left_x_new, right_x_new):
        """
        :param left_x_new:
        :param right_x_new:
        :return:
        """
        # print(np.sum(CurvatureParams.left_lane_n_polynomial_matrix, axis=0))
        if CurvatureParams.running_index >= 9:
            left_x_new = np.sum(
                    CurvatureParams.left_lane_n_polynomial_matrix * CurvatureParams.moving_average_weigths, axis=1
            )
            right_x_new = np.sum(
                    CurvatureParams.right_lane_n_polynomial_matrix * CurvatureParams.moving_average_weigths, axis=1
            )
        return left_x_new, right_x_new

    @staticmethod
    def check_lines_with_variance_of_curvature_change(left_x_new, right_x_new):
        
        if CurvatureParams.running_index > 0:
            prev_left_lane_poly_x = CurvatureParams.left_lane_n_polynomial_matrix[::, -1]
            prev_right_lane_poly_x = CurvatureParams.right_lane_n_polynomial_matrix[::, -1]
           
            prev_left_lane_poly_x = (prev_left_lane_poly_x - np.mean(prev_left_lane_poly_x)) / (np.std(
                    prev_left_lane_poly_x))
            prev_right_lane_poly_x = (prev_right_lane_poly_x - np.mean(prev_right_lane_poly_x)) / (np.std(
                    prev_right_lane_poly_x))

            curr_left_lane_poly_x = (left_x_new - np.mean(left_x_new)) / np.std(left_x_new)
            curr_right_lane_poly_x = (right_x_new - np.mean(right_x_new)) / np.std(right_x_new)
            
            left_lane_variance = np.var(curr_left_lane_poly_x - prev_left_lane_poly_x)
            right_lane_variance = np.var(curr_right_lane_poly_x - prev_right_lane_poly_x)
            
            CurvatureParams.left_line_curr_poly_variance += [left_lane_variance]
            CurvatureParams.right_line_curr_poly_variance += [right_lane_variance]
            
            if np.isnan(left_lane_variance):
                raise ValueError("Oops you got a np.nan")
            if np.isnan(right_lane_variance):
                raise ValueError("Oops you got a np.nan")
            if left_lane_variance > CurvatureParams.allowed_variance:
                left_x_new = CurvatureParams.left_lane_n_polynomial_matrix[::, -1]
                
            if right_lane_variance > CurvatureParams.allowed_variance:
                right_x_new = CurvatureParams.right_lane_n_polynomial_matrix[::, -1]
        return left_x_new, right_x_new

    @staticmethod
    def set_curr_left_lane_points(left_x_new, left_y_new):
        # Overwrite the predicted points as new search space
        CurvatureParams.left_lane_x_points = left_x_new
        CurvatureParams.left_lane_y_points = left_y_new
    
        # print(f'\n[Frame Curvature Points]\n'
        #       f'left_lane_y_points = {len(CurvatureParams.left_lane_y_points)}, '
        #       f'left_lane_x_points = {len(CurvatureParams.left_lane_x_points)} ')

    @staticmethod
    def set_curr_right_lane_points(right_x_new, right_y_new):
        CurvatureParams.right_lane_x_points = right_x_new
        CurvatureParams.right_lane_y_points = right_y_new
    
        # print(f'\n[Frame Curvature Points]\n'
        #       f'right_lane_y_points = {len(CurvatureParams.right_lane_y_points)}, '
        #       f'right_lane_x_points = {len(CurvatureParams.right_lane_x_points)} ')
        
    @staticmethod
    def set_curr_lane_points_on_polynomial_matrix(left_x_new, right_x_new):
        """
        Here we assign the new left_right lane points to [:, nth] column position
        :param left_x_new:
        :param right_x_new:
        :return:
        """
        CurvatureParams.left_lane_n_polynomial_matrix[:, -1:] = left_x_new.reshape(-1, 1)
        CurvatureParams.right_lane_n_polynomial_matrix[:, -1:] = right_x_new.reshape(-1, 1)
        
    @staticmethod
    def shift_polynomial_matrix():
        """
        Here we shift the lane points [:, 1:n] to [:, 0:n-1]
        :return:
        """
        CurvatureParams.left_lane_n_polynomial_matrix[:, :-1] = CurvatureParams.left_lane_n_polynomial_matrix[:, 1:]
        CurvatureParams.right_lane_n_polynomial_matrix[:, :-1] = CurvatureParams.right_lane_n_polynomial_matrix[:, 1:]
        


"""
1. Find colorspace and gradients Dynamically change
2.
"""
# TODO Take points in polynomial they should converge as the lane move upwards in the image
# When the lane diverges a lot then use sliding window technique
# When doing sliding window

    