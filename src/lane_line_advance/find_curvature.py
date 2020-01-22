import cv2
import numpy as np


def birds_eye_view(img, src_points, dst_points):
    """
    :param img:
    :param src_points: list(tuples)) or nd_array (4, 2)
    :param dst_points:
    :return:
    """
    h, w, _ = img.shape
    src_points = np.array(src_points).astype(np.float32)
    dst_points = np.array(dst_points).astype(np.float32)
    
    assert(src_points.shape[1] == 2 and src_points.shape[0] >= 4)
    assert(dst_points.shape[1] == 2 and dst_points.shape[0] >= 4)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST)
    return warped_image


def fetch_start_position_with_hist_dist(preprocessed_bin_image, save_path=None):
    """
    At this point we should have a binary image (1, 0) with detected lane line. Here detections are annotated with 1
    This method is necessary to provide a starting point to find the curvature of the lane line
    :param preprocessed_image:
    :param save_path:
    :return:
        left_lane_start_point_coordinates = (y1, x1)
        right_lane_start_point_coordinates = (y2, x2)
    # TODO: Lane Line are broad and hence the values between the edges of line lines are 0. It would be a good
    practise to apply a kernel to fill up these valleys
    # TODO: Remove Outliers (Bad Gradients)
    # TODO: How we we handle Bimodal Distribution for a particular lane
    # TODO: Can we try weighted method (Something like a moving Average)
    # TODO: Should we try sum of multiple neighboring columns instead of using just one.
    """
    # preprocessed_image[preprocessed_image > 0] = 1
    assert(set(np.unique(preprocessed_bin_image)) == {0, 1}), (
        f'The preprocessed image should be binary {0, 1} but contains values {set(np.unique(preprocessed_bin_image))}'
    )
    # Sum all the values in column axis
    frequency_histogram = np.sum(preprocessed_bin_image, axis=0)
    # Divide the Frequency histogram into two parts to find starting points for Left Lane and Right Lane
    left_lane = frequency_histogram[0: len(frequency_histogram) // 2]
    right_lane = frequency_histogram[len(frequency_histogram) // 2:]
    
    if save_path:
        fig = commons.basic_plot(nrows=1, ncols=3, figsize=(50, 10))(
                [frequency_histogram, left_lane, right_lane],
                ["frequency_histogram", "hist_left_lane", "hist_right_lane"]
        )
        commons.save_matplotlib(save_path, fig)
    
    left_lane_start_index = np.argmax(left_lane)
    right_lane_start_index = len(frequency_histogram) // 2 + np.argmax(right_lane)
    print(left_lane_start_index, right_lane_start_index)
    return (preprocessed_bin_image.shape[0], left_lane_start_index), (preprocessed_bin_image.shape[0], right_lane_start_index)
    
    
class LaneCurvature:
    def __init__(self, preprocessed_bin_image, left_lane_pos_yx, right_lane_pos_yx, window_size, save_path):
        self.preprocessed_bin_image = preprocessed_bin_image
        self.left_lane_pos_yx = left_lane_pos_yx
        self.right_lane_pos_yx = right_lane_pos_yx
        self.window_size = window_size
        self.save_path = save_path
        self.left_lane_y_points = []
        self.left_lane_x_points = []
        self.right_lane_y_points = []
        self.right_lane_x_points = []

        self.left_fit = None
        self.right_fit = None
        
        if save_path:
            self.preprocessed_image_cp = np.dstack((
                self.preprocessed_bin_image, self.preprocessed_bin_image, self.preprocessed_bin_image
            ))
            self.preprocessed_image_cp[self.preprocessed_bin_image == 1] = 255
    
    def plot_rectangles(self, left_box, right_box):
        cv2.rectangle(
            self.preprocessed_image_cp,
            (left_box[1], left_box[0]), (left_box[3], left_box[2]),
            (255, 0, 0), 2
        )
    
        cv2.rectangle(
            self.preprocessed_image_cp,
            (right_box[1], right_box[0]), (right_box[3], right_box[2]),
            (255, 0, 0), 2
        )
    
    def store_curvature_points(self, left_box, right_box):
        # TODO: This process can be initiated by a new thread since this is just storage
        left_box_vals = self.preprocessed_bin_image[left_box[0]:left_box[2], left_box[1]:left_box[3]]
        right_box_vals = self.preprocessed_bin_image[right_box[0]:right_box[2], right_box[1]:right_box[3]]
        
        ll_non_zero_y_idx, ll_non_zero_x_idx = np.where(left_box_vals == 1)
        rl_non_zero_y_idx, rl_non_zero_x_idx = np.where(right_box_vals == 1)

        self.left_lane_y_points += list(ll_non_zero_y_idx + left_box[0])
        self.left_lane_x_points += list(ll_non_zero_x_idx + left_box[1])
        self.right_lane_y_points += list(rl_non_zero_y_idx + right_box[0])
        self.right_lane_x_points += list(rl_non_zero_x_idx + right_box[1])
        
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
        
    def find_lane_points(self):
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
            
            if self.save_path:
                self.plot_rectangles(left_box, right_box)
                
            cnt += 1
        
        if self.save_path:
            self.preprocessed_image_cp = np.array(self.preprocessed_image_cp).astype(np.uint8)
            commons.save_image(self.save_path, self.preprocessed_image_cp)

    def fit(self, degree=2):
        self.left_fit = np.polyfit(self.left_lane_y_points, self.left_lane_x_points, deg=degree)
        self.right_fit = np.polyfit(self.right_lane_y_points, self.right_lane_x_points, deg=degree)
        
    def predict(self):
        """
        y = b0 + b1*x, b2*x_sq
        :return:
        """
        y_new = np.linspace(0, self.preprocessed_bin_image.shape[0] - 1, self.preprocessed_bin_image.shape[0])
        try:
            left_x_new = self.left_fit[0] * y_new ** 2 + self.left_fit[1] * y_new + self.left_fit[2]
            right_x_new = self.right_fit[0] * y_new ** 2 + self.right_fit[1] * y_new + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_x_new = 1 * y_new ** 2 + 1 * y_new
            right_x_new = 1 * y_new ** 2 + 1 * y_new
        return y_new, left_x_new, right_x_new
    
    def plot(self, left_y_new, left_x_new, right_y_new, right_x_new):
        draw_points = np.asarray([left_x_new, left_y_new]).T.astype(np.int32)
        cv2.polylines(self.preprocessed_image_cp, [draw_points], False, (50, 255, 255), thickness=4)
        draw_points = np.asarray([right_x_new, right_y_new]).T.astype(np.int32)
        cv2.polylines(self.preprocessed_image_cp, [draw_points], False, (50, 255, 255), thickness=4)

        if self.save_path:
            self.preprocessed_image_cp = np.array(self.preprocessed_image_cp).astype(np.uint8)
            commons.save_image(self.save_path, self.preprocessed_image_cp)

from src import commons
from src.lane_line_advance.main import PreprocessingPipeline

test_image_name = "test4"#"straight_lines1"  # test4

input_image_path = f'./data/test_images/{test_image_name}.jpg'
output_plot_path = f'./data/output_images/{test_image_name}.png'
hist_output_path = f"./data/output_images/histogram/{test_image_name}.png"
curvature_bbox_output_path = f"./data/output_images/curvature_{test_image_name}.png"

image = commons.read_image(input_image_path)
pp_pipeline = PreprocessingPipeline(image, save_path=output_plot_path)
pp_pipeline.warp()
preprocessed_bin_image = pp_pipeline.preprocess()
preprocessed_bin_image = preprocessed_bin_image.astype(np.int32)
print('preprocessed_img: ', np.unique(preprocessed_bin_image))

left_lane_pos_yx, right_lane_pos_yx = fetch_start_position_with_hist_dist(
        preprocessed_bin_image.copy(), save_path=hist_output_path
)

obj_l_curv = LaneCurvature(
    preprocessed_bin_image=preprocessed_bin_image,
    left_lane_pos_yx=left_lane_pos_yx,
    right_lane_pos_yx=right_lane_pos_yx,
    window_size=(70, 130),
    save_path=curvature_bbox_output_path
)
obj_l_curv.find_lane_points()
obj_l_curv.fit()
y_new, left_x_new, right_x_new = obj_l_curv.predict()
obj_l_curv.plot(y_new, left_x_new, y_new, right_x_new)


    
    