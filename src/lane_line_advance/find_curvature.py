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
    
    
def find_lane_curvature(preprocessed_bin_image, left_lane_pos_yx, right_lane_pos_yx, filter_shape, save_path):
    print('leftlane rightlane: ', left_lane_pos_yx, right_lane_pos_yx)
    ll_y_mid_point = left_lane_pos_yx[0] - (filter_shape[0]//2)
    ll_x_mid_point = left_lane_pos_yx[1]
    print(ll_y_mid_point, ll_x_mid_point)

    rl_y_mid_point = right_lane_pos_yx[0] - (filter_shape[0] // 2)
    rl_x_mid_point = right_lane_pos_yx[1]
    print(rl_y_mid_point, rl_x_mid_point)
    
    cnt = 4
    while cnt <= 5:
        left_box = [
            ll_y_mid_point - filter_shape[0] // 2, ll_x_mid_point - filter_shape[1] // 2,
            ll_y_mid_point + filter_shape[0] // 2, ll_x_mid_point + filter_shape[1] // 2,
        ]
        right_box = [
            rl_y_mid_point - filter_shape[0] // 2, rl_x_mid_point - filter_shape[1] // 2,
            rl_y_mid_point + filter_shape[0] // 2, rl_x_mid_point + filter_shape[1] // 2,
        ]
        print('preprocessed_image: ', preprocessed_bin_image.shape)
        print('leftbox and right box: ', left_box, right_box)
        left_box_vals = preprocessed_bin_image[left_box[0]:left_box[2], left_box[1]:left_box[3]]
        right_box_vals = preprocessed_bin_image[right_box[0]:right_box[2], right_box[1]:right_box[3]]
        print(left_box_vals)
        ll_max_pos = np.argmax(left_box_vals)
        rl_max_pos = np.argmax(right_box_vals)

        preprocessed_image_cp = np.dstack((preprocessed_bin_image, preprocessed_bin_image, preprocessed_bin_image))
        preprocessed_image_cp[preprocessed_bin_image == 1] = 255
        print('2222222222222222: ', np.unique(preprocessed_image_cp))
        cv2.rectangle(
                preprocessed_image_cp,
                (left_box[1], left_box[0]), (left_box[3], left_box[2]),
                (255, 0, 0), 2
        )

        cv2.rectangle(
                preprocessed_image_cp,
                (right_box[1], right_box[0]), (right_box[3], right_box[2]),
                (255, 0, 0), 2
        )
        print('Maximum Position: ', ll_max_pos, rl_max_pos)
        cnt += 1

    if save_path:
        # preprocessed_image_cp = np.array(preprocessed_image_cp).astype(np.uint8)
        print('42423423423432: ', np.unique(preprocessed_image_cp))
        commons.save_image(save_path, preprocessed_image_cp)
    
    
    
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
find_lane_curvature(
        preprocessed_bin_image=preprocessed_bin_image,
        left_lane_pos_yx=left_lane_pos_yx,
        right_lane_pos_yx=right_lane_pos_yx,
        filter_shape=(70, 130),
        save_path=curvature_bbox_output_path
)

# pp_pipeline.plot()


    
    