import cv2
import numpy as np

from src import commons
from src.lane_line_advance.preprocess import Preprocess


class BasePipeline:
    def __init__(self, image):
        self.image = image
        
        self.warped_image = None
        self.h, self.w, _ = self.image.shape
        
        self.src_points = [(500, 450), (0, 700), (750, 450), (1250, 700)]
        self.dst_points = [(0, 0), (0, self.h - 1), (self.w - 1, 0), (self.w - 1, self.h - 1)]


class PreprocessingPipeline(BasePipeline):
    def __init__(self, image, save_path=None):
        super().__init__(image)
        self.save_path = save_path
        if save_path is not None:
            self.plot_images = [image]
            self.plot_names = ["orig_image"]
    
    def warp(self):
        if self.save_path:
            out_img = self.image.copy()
            lineThickness = 2
            cv2.line(out_img, self.src_points[0], self.src_points[1], (0, 255, 0), lineThickness)
            cv2.line(out_img, self.src_points[2], self.src_points[3], (0, 255, 0), lineThickness)
            self.plot_images += [out_img]
            self.plot_names += ["warp_region"]
        
        M = cv2.getPerspectiveTransform(
                np.array(self.src_points).astype(np.float32), np.array(self.dst_points).astype(np.float32)
        )
        
        self.warped_image = cv2.warpPerspective(self.image.copy(), M, (self.w, self.h), flags=cv2.INTER_NEAREST)
        
        if self.save_path:
            self.plot_images += [self.warped_image]
            self.plot_names += ["warped_image"]
            
    def threshold(self, channel, thresh=(15, 50)):
        binary = np.zeros_like(channel)
        binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
        return binary
    
    def preprocess(self):
        r, g, b = [np.squeeze(i, axis=2) for i in np.dsplit(self.warped_image, 3)]
        obj_pp = Preprocess(image=self.warped_image)
        hls_img = obj_pp.apply_colorspace(cv2.COLOR_RGB2HLS)

        h, l, s = [np.squeeze(i, axis=2) for i in np.dsplit(hls_img, 3)]

        # Capture R and S channel that are most effective
        binary_r = self.threshold(r, (150, 255))
        binary_s = self.threshold(s, (15, 150))

        # Get logical OR between R and S Channel
        rs_active_pxl = np.logical_and(binary_r, binary_s)#binary_r + binary_s
        rs_active_pxl[rs_active_pxl > 0] = 1

        # Get Gradients on RBG->BLUR->Gray color space and Apply Absolute Gradient thresholding
        ls_img = np.dstack([l, s, np.zeros(h.shape)]).astype(np.uint8)
        obj_pp.reset_image(ls_img)
        ls_gray = obj_pp.apply_colorspace(cv2.COLOR_RGB2GRAY)

        gx, gy = obj_pp.apply_gradients(kernel_size=3)
        x_abs_thresh_img = obj_pp.apply_absolute_thresh(axis="x", threshold=(15, 150))

        # Perform Logical AND between absolute_gradients and RS active channels
        preprocessed_img = np.logical_and(x_abs_thresh_img, rs_active_pxl)

        if self.save_path is not None:
            binary_g = self.threshold(g, (150, 255))
            binary_b = self.threshold(b, (15, 50))
            binary_h = self.threshold(h, (10, 50))
            binary_l = self.threshold(l, (150, 255))

            obj_pp.reset_image(self.warped_image.copy())
            obj_pp.apply_blurr(kernel=3)
            gray_img = obj_pp.apply_colorspace(cv2.COLOR_RGB2GRAY)

            self.plot_images += [
                r, g, b,
                binary_r, binary_g, binary_b,
                h, l, s,
                binary_h, binary_l, binary_s,
                rs_active_pxl, gray_img, ls_gray,
                gx, x_abs_thresh_img, preprocessed_img
            ]
            self.plot_names += [
                "red", "green", "blue",
                "binary_r", "binary_g", "binary_b",
                "hue", "lightning", "saturation",
                "binary_h", "binary_l", "binary_s",
                "r_or_s_colorspace", "gray_colorsapce", "ls_gray_colorspace",
                "ls_gray_gradient_y", "ls_gray_gradient_y_abs_thres", "preprocessed_img"
            ]

        return preprocessed_img

    def preprocess(self):
        r, g, b = [np.squeeze(i, axis=2) for i in np.dsplit(self.warped_image, 3)]
        obj_pp = Preprocess(image=self.warped_image)
        hls_img = obj_pp.apply_colorspace(cv2.COLOR_RGB2HLS)
    
        h, l, s = [np.squeeze(i, axis=2) for i in np.dsplit(hls_img, 3)]
    
        # Capture R and S channel that are most effective
        binary_r = self.threshold(r, (150, 255))
        binary_s = self.threshold(s, (15, 150))
    
        # Get logical OR between R and S Channel
        rs_active_pxl = np.logical_and(binary_r, binary_s)  # binary_r + binary_s
        rs_active_pxl[rs_active_pxl > 0] = 1
    
        # Get Gradients on RBG->BLUR->Gray color space and Apply Absolute Gradient thresholding
        ls_img = np.dstack([l, s, np.zeros(h.shape)]).astype(np.uint8)
        obj_pp.reset_image(s)
        # ls_gray = obj_pp.apply_colorspace(cv2.COLOR_RGB2GRAY)
    
        gx, gy = obj_pp.apply_gradients(kernel_size=3)
        x_abs_thresh_img = obj_pp.apply_absolute_thresh(axis="x", threshold=(15, 150))
    
        # Perform Logical AND between absolute_gradients and RS active channels
        preprocessed_img = np.logical_and(x_abs_thresh_img, rs_active_pxl)
    
        if self.save_path is not None:
            binary_g = self.threshold(g, (150, 255))
            binary_b = self.threshold(b, (15, 50))
            binary_h = self.threshold(h, (10, 50))
            binary_l = self.threshold(l, (150, 255))
        
            obj_pp.reset_image(self.warped_image.copy())
            obj_pp.apply_blurr(kernel=3)
            gray_img = obj_pp.apply_colorspace(cv2.COLOR_RGB2GRAY)
        
            self.plot_images += [
                r, g, b,
                binary_r, binary_g, binary_b,
                h, l, s,
                binary_h, binary_l, binary_s,
                rs_active_pxl, gray_img,#, ls_gray,
                gx, x_abs_thresh_img, preprocessed_img
            ]
            self.plot_names += [
                "red", "green", "blue",
                "binary_r", "binary_g", "binary_b",
                "hue", "lightning", "saturation",
                "binary_h", "binary_l", "binary_s",
                "r_or_s_colorspace", "gray_colorsapce",#, "ls_gray_colorspace",
                "ls_gray_gradient_y", "ls_gray_gradient_y_abs_thres", "preprocessed_img"
            ]
    
        return preprocessed_img
    
    def plot(self):
        assert (len(self.plot_images) == len(self.plot_names))
        ncol = 3
        nrows = int(np.ceil(len(self.plot_names) / ncol))
        fig = commons.image_subplots(nrows=nrows, ncols=ncol)(self.plot_images, self.plot_names)
        commons.save_matplotlib(self.save_path, fig)


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
                str((left_lane_curvature_radius/1000).round(3)), pos=(250, 250), color=(255, 0, 0)
        )
        self.obj_img_plots.add_caption(
                str((right_lane_curvature_radius/1000).round(3)), pos=(950, 250), color=(255, 0, 0)
        )
        return self.obj_img_plots.image
    
    def plot(self):
        assert (len(self.plot_images) == len(self.plot_names))
        ncol = 3
        nrows = int(np.ceil(len(self.plot_names) / ncol))
        fig = commons.image_subplots(nrows=nrows, ncols=ncol)(self.plot_images, self.plot_names)
        commons.save_matplotlib(self.save_path, fig)
