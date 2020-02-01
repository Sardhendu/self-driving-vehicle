import cv2
import numpy as np

from src import commons
from src.lane_line_advance.preprocess import Preprocess
from src.lane_line_advance.lane_curvature import get_variance_of_curvature_change
from src.lane_line_advance.lane_curvature import LaneCurvature, fetch_start_position_with_hist_dist
from src.lane_line_advance.params import PipelineParams, CurvatureParams
    

class PreprocessBuilder:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        self.plot_images = []
        self.plot_names = []
    
    def warp(self, image):
        if self.save_dir:
            img_copy = image.copy()
            lineThickness = 2
            cv2.line(img_copy, PipelineParams.src_points[0], PipelineParams.src_points[1], (0, 255, 0), lineThickness)
            cv2.line(img_copy, PipelineParams.src_points[2], PipelineParams.src_points[3], (0, 255, 0), lineThickness)
            self.plot_images += [img_copy]
            self.plot_names += ["warp_region"]
        
        image = cv2.warpPerspective(
                image.copy(), PipelineParams.M, (PipelineParams.width, PipelineParams.height), flags=cv2.INTER_NEAREST
        )
        
        if self.save_dir:
            self.plot_images += [image]
            self.plot_names += ["warped_image"]
            self.plot(ncol=2, save_path=f"{self.save_dir}/perspective_transform.png")
            self.plot_images, self.plot_names = [], []
        return image
    
    def threshold(self, channel, thresh=(15, 50)):
        binary = np.zeros_like(channel)
        binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
        return binary
    
    def preprocess(self, image, threshold_index):
        r, g, b = [np.squeeze(i, axis=2) for i in np.dsplit(image, 3)]
        obj_pp = Preprocess(image=image.copy())
        hls = obj_pp.apply_colorspace(cv2.COLOR_RGB2HLS)
        
        h, l, s = [np.squeeze(i, axis=2) for i in np.dsplit(hls, 3)]
        
        # Capture R and S channel that are most effective
        binary_r = self.threshold(r, PipelineParams.red_threshold[threshold_index])
        binary_s = self.threshold(s, PipelineParams.saturation_threshold[threshold_index])  # More that 15 is difficult
        print(np.sum(binary_s))
        
        # Get logical OR between R and S Channel
        r_or_s = np.logical_or(binary_r, binary_s)  # binary_r + binary_s
        r_or_s[r_or_s > 0] = 1
        
        r_and_s = np.logical_and(binary_r, binary_s)  # binary_r + binary_s
        r_and_s[r_and_s > 0] = 1
        
        # Get Gradients on RBG->BLUR->Gray color space and Apply Absolute Gradient threshold
        ls = np.dstack([l, s, r]).astype(np.uint8)
        obj_pp.reset_image(ls)
        out_gray = obj_pp.apply_colorspace(cv2.COLOR_RGB2GRAY)
        gx, gy = obj_pp.apply_gradients(kernel_size=3)
        x_abs_thresh_img = obj_pp.apply_absolute_thresh(
                axis="x", threshold=PipelineParams.gradient_threshold[threshold_index]
        )
        
        # Perform Logical AND between absolute_gradients and RS active channels
        preprocessed_img = np.logical_and(x_abs_thresh_img, r_and_s).astype(np.uint8)
        
        # Only used while debugging
        if self.save_dir is not None:
            binary_g = self.threshold(g, (150, 255))
            binary_b = self.threshold(b, (15, 50))
            binary_h = self.threshold(h, (10, 50))
            binary_l = self.threshold(l, (150, 255))

            obj_pp.reset_image(image.copy())
            obj_pp.apply_blurr(kernel=3)
            rgb2gray = obj_pp.apply_colorspace(cv2.COLOR_RGB2GRAY)
            
            obj_pp.reset_image(hls.copy())
            obj_pp.apply_blurr(kernel=3)
            hls2gray = obj_pp.apply_colorspace(cv2.COLOR_RGB2GRAY)

            self.plot_images += [
                image.copy(), hls, rgb2gray, hls2gray,
                r, g,
                binary_r, binary_g, binary_b,
                h, l, s,
                binary_h, binary_l, binary_s,
                r_and_s, r_or_s, out_gray,
                gx, x_abs_thresh_img, preprocessed_img
            ]
            self.plot_names += [
                "orig_image", "hls", "rgb2gray", "hls2gray",
                "red", "green",
                "binary_r", "binary_g", "binary_b",
                "hue", "lightning", "saturation",
                "binary_h", "binary_l", "binary_s",
                "r_and_s", "r_or_s", "used_gray_img",
                "ls2gray_gradY", "ls2gray_gradY_abs_thres", "preprocessed_img"
            ]
            self.plot(ncol=3, save_path=f"{self.save_dir}/preprocessed_image.png")
            self.plot_images = []
            self.plot_names = []

            color_binary = np.dstack((np.zeros_like(x_abs_thresh_img), x_abs_thresh_img, binary_s)) * 255
            cv2.imwrite("11.png",color_binary)
        return preprocessed_img
    
    def plot(self, ncol, save_path):
        assert (len(self.plot_images) == len(self.plot_names)), (f'{len(self.plot_images)} != {len(self.plot_names)}')
        nrows = int(np.ceil(len(self.plot_names) / ncol))
        fig = commons.image_subplots(nrows=nrows, ncols=ncol)(self.plot_images, self.plot_names)
        commons.save_matplotlib(save_path, fig)

    
class PostprocessingBuilder:
    def __init__(self, image, save_dir=None):
        self.image = image
        self.obj_img_plots = commons.ImagePlots(image)
        
        self.save_dir = save_dir
        if save_dir is not None:
            self.plot_images = [image]
            self.plot_names = ["orig_image"]
    
    def transform_lane_points(self, left_lane_points: np.array, right_lane_points: np.array):
        cnt_left_lane_pnts = len(left_lane_points)
        cnt_right_lane_pnts = len(right_lane_points)
        ones = np.ones(cnt_left_lane_pnts + cnt_right_lane_pnts).reshape(-1, 1)
        input_points = np.column_stack((
            np.vstack((left_lane_points, right_lane_points)), ones
        ))
        
        transformed_points = np.dot(PipelineParams.M_inv, input_points.T).T
        dividor = transformed_points[:, -1].reshape(-1, 1)
        transformed_points = transformed_points[:, 0:2]
        transformed_points /= dividor
        transformed_points = transformed_points.astype(np.int32)
        left_lane = transformed_points[0:cnt_left_lane_pnts]
        right_lane = transformed_points[cnt_left_lane_pnts:]
        
        return left_lane, right_lane
    
    def draw_lane_mask(self, left_lane, right_lane, left_lane_curvature_radius, right_lane_curvature_radius):
        if self.save_dir:
            unwarped_image = cv2.warpPerspective(
                    self.obj_img_plots.image.copy(), PipelineParams.M_inv,
                    (PipelineParams.width, PipelineParams.height), flags=cv2.INTER_NEAREST
            )
            self.obj_img_plots.polymask(
                    points=np.vstack((left_lane, right_lane[::-1])),
                    color=(0, 255, 0),
                    mask_weight=0.5
            )
            self.plot_images += [unwarped_image, self.obj_img_plots.image]
            self.plot_names += ["unwarped_image", "transformed_lane_points"]
        
        self.obj_img_plots.polymask(
                points=np.vstack((left_lane, right_lane[::-1])),
                color=(0, 255, 0),
                mask_weight=0.5
        )
        self.obj_img_plots.add_caption(
                str((left_lane_curvature_radius / 1000).round(3)), pos=(250, 250), color=(255, 0, 0)
        )
        self.obj_img_plots.add_caption(
                str((right_lane_curvature_radius / 1000).round(3)), pos=(950, 250), color=(255, 0, 0)
        )
        return self.obj_img_plots.image
    
    def plot(self):
        assert (len(self.plot_images) == len(self.plot_names))
        ncol = 3
        nrows = int(np.ceil(len(self.plot_names) / ncol))
        fig = commons.image_subplots(nrows=nrows, ncols=ncol)(self.plot_images, self.plot_names)
        commons.save_matplotlib(f'{self.save_dir}/postprocess.png', fig)


def preprocessing_pipeline(image, threshold_index, save_dir=None):
    preprocess_pipeline = PreprocessBuilder(save_dir)
    preprocessed_img = preprocess_pipeline.preprocess(image, threshold_index)
    preprocessed_bin_image = preprocess_pipeline.warp(preprocessed_img)
    preprocessed_bin_image = preprocessed_bin_image.astype(np.int32)
    return preprocessed_bin_image


def lane_curvature_pipeline(preprocessed_bin_image, save_dir, mode):
    # -------------------------------------------------------------------------------------
    # Get histogram distribution to determine start point for sliding window
    # -------------------------------------------------------------------------------------
    left_lane_pos_yx, right_lane_pos_yx = fetch_start_position_with_hist_dist(
            preprocessed_bin_image.copy(), save_dir=save_dir
    )
    
    lane_curvature = LaneCurvature(
            preprocessed_bin_image=preprocessed_bin_image,
            left_lane_pos_yx=left_lane_pos_yx,
            right_lane_pos_yx=right_lane_pos_yx,
            margin=100,
            save_dir=save_dir,
            pipeline=mode
    )
    lane_curvature.find_lane_points()
    lane_curvature.fit()
    y_new, left_x_new, right_x_new = lane_curvature.predict()
    
    if mode != "debug":
        left_lane_variance, right_lane_variance = get_variance_of_curvature_change(
                np.column_stack((y_new, left_x_new)), np.column_stack((y_new, right_x_new))
        )
        
    if mode == "warped":
        print(f'\n[Lane Curvature] '
              f'y_new = {len(y_new)}, left_x_new = {len(left_x_new)}, right_x_new = {len(right_x_new)}')
        lane_curvature.plot(y_new, left_x_new, y_new, right_x_new)
        return lane_curvature.preprocessed_img_plot.image
    
    lane_curvature.measure_radius_in_meter()
    return left_x_new, right_x_new, y_new
    
    
def postprocessing_pipeline(image, left_x_new, right_x_new, y_new, save_dir, mode):
    postprocess_pipeline = PostprocessingBuilder(image, save_dir=save_dir)
    left_lane_points, right_lane_points = postprocess_pipeline.transform_lane_points(
            left_lane_points=np.column_stack((left_x_new, y_new)),
            right_lane_points=np.column_stack((right_x_new, y_new))
    )
    
    if mode == "debug":
        print(
                f'\n[Lane Detection] '
                f'left_lane = {len(left_lane_points)}, right_lane = {len(right_lane_points)}'
        )
        
    out_image = postprocess_pipeline.draw_lane_mask(
            left_lane_points, right_lane_points,
            CurvatureParams.left_lane_curvature_radii_curr,
            CurvatureParams.right_lane_curvature_radii_curr
    )
    if mode == "debug":
        postprocess_pipeline.plot()
        
    return out_image
