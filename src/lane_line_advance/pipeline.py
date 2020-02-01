import cv2
import numpy as np

from src import commons
from src.lane_line_advance.preprocess import Preprocess


class BasePipeline:
    def __init__(self, image):
        self.image = image
        
        self.warped_image = None
        self.h, self.w, _ = self.image.shape
        
        self.src_points = [(500, 450), (0, 700), (780, 450), (1280, 700)]
        # self.src_points = [(500, 450), (0, 700), (750, 450), (1250, 700)]
        self.dst_points = [(0, 0), (0, self.h - 1), (self.w - 1, 0), (self.w - 1, self.h - 1)]


class PreprocessingPipeline(BasePipeline):
    def __init__(self, image, save_dir=None):
        super().__init__(image)
        self.save_dir = save_dir
        if save_dir is not None:
            self.orig_img = self.image.copy()
            self.plot_images = [image]
            self.plot_names = ["orig_image"]
    
    def warp(self, image):
        if self.save_dir:
            img_copy = self.image.copy()
            lineThickness = 2
            cv2.line(img_copy, self.src_points[0], self.src_points[1], (0, 255, 0), lineThickness)
            cv2.line(img_copy, self.src_points[2], self.src_points[3], (0, 255, 0), lineThickness)
            self.plot_images += [img_copy]
            self.plot_names += ["warp_region"]
        
        M = cv2.getPerspectiveTransform(
                np.array(self.src_points).astype(np.float32), np.array(self.dst_points).astype(np.float32)
        )
        image = cv2.warpPerspective(image.copy(), M, (self.w, self.h), flags=cv2.INTER_NEAREST)
        
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
    
    def preprocess(self, image):
        r, g, b = [np.squeeze(i, axis=2) for i in np.dsplit(image, 3)]
        obj_pp = Preprocess(image=image.copy())
        hls = obj_pp.apply_colorspace(cv2.COLOR_RGB2HLS)
        
        h, l, s = [np.squeeze(i, axis=2) for i in np.dsplit(hls, 3)]
        
        # Capture R and S channel that are most effective
        binary_r = self.threshold(r, (150, 255))
        binary_s = self.threshold(s, (100, 255))  # More that 15 is difficult
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
       
        # obj_pp.reset_image(l)
        gx, gy = obj_pp.apply_gradients(kernel_size=3)
        x_abs_thresh_img = obj_pp.apply_absolute_thresh(axis="x", threshold=(15, 100))
        
        # Perform Logical AND between absolute_gradients and RS active channels
        preprocessed_img = np.logical_and(x_abs_thresh_img, r_and_s).astype(np.uint8)
        
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
                hls, rgb2gray, hls2gray,
                r, g,
                binary_r, binary_g, binary_b,
                h, l, s,
                binary_h, binary_l, binary_s,
                r_and_s, r_or_s, out_gray,
                gx, x_abs_thresh_img, preprocessed_img
            ]
            self.plot_names += [
                "hls", "rgb2gray", "hls2gray",
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


class PostprocessingPipeline(BasePipeline):
    def __init__(self, image, save_path=None):
        super().__init__(image)
        self.M = None
        self.obj_img_plots = commons.ImagePlots(image)
        
        self.save_path = save_path
        if save_path is not None:
            self.plot_images = [image]
            self.plot_names = ["orig_image"]
    
    def unwarp(self):
        src_points = self.dst_points
        dst_points = self.src_points
        self.M = cv2.getPerspectiveTransform(
                np.array(src_points).astype(np.float32), np.array(dst_points).astype(np.float32)
        )
    
    def transform_lane_points(self, left_lane_points: np.array, right_lane_points: np.array):
        cnt_left_lane_pnts = len(left_lane_points)
        cnt_right_lane_pnts = len(right_lane_points)
        ones = np.ones(cnt_left_lane_pnts + cnt_right_lane_pnts).reshape(-1, 1)
        input_points = np.column_stack((
            np.vstack((left_lane_points, right_lane_points)), ones
        ))
        
        transformed_points = np.dot(self.M, input_points.T).T
        dividor = transformed_points[:, -1].reshape(-1, 1)
        transformed_points = transformed_points[:, 0:2]
        transformed_points /= dividor
        transformed_points = transformed_points.astype(np.int32)
        left_lane = transformed_points[0:cnt_left_lane_pnts]
        right_lane = transformed_points[cnt_left_lane_pnts:]
        
        return left_lane, right_lane
    
    def draw_lane_mask(self, left_lane, right_lane, left_lane_curvature_radius, right_lane_curvature_radius):
        if self.save_path:
            unwarped_image = cv2.warpPerspective(
                    self.obj_img_plots.image.copy(), self.M, (self.w, self.h), flags=cv2.INTER_NEAREST
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
        commons.save_matplotlib(self.save_path, fig)

