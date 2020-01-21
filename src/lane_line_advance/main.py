
import cv2
import numpy as np

from src import commons
from src.lane_line_advance.preprocess import Preprocess


class PreprocessingPipeline:
    def __init__(self, image, save_path=None):
        self.image = image
        self.save_path = save_path
        self.warped_image = None
        if save_path is not None:
            self.plot_images = [image]
            self.plot_names = ["orig_image"]
        
    def warp(self, src_points=[(500, 450), (0, 700), (750, 450), (1250, 700)]):
        if self.save_path:
            out_img = self.image.copy()
            lineThickness = 2
            cv2.line(out_img, src_points[0], src_points[1], (0, 255, 0), lineThickness)
            cv2.line(out_img, src_points[2], src_points[3], (0, 255, 0), lineThickness)
            self.plot_images += [out_img]
            self.plot_names += ["warp_region"]

        h, w, _ = self.image.shape
        dst_points = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]
        M = cv2.getPerspectiveTransform(
                np.array(src_points).astype(np.float32), np.array(dst_points).astype(np.float32)
        )
        self.warped_image = cv2.warpPerspective(self.image.copy(), M, (w, h), flags=cv2.INTER_NEAREST)
        
        if self.save_path:
            self.plot_images += [self.warped_image]
            self.plot_names += ["warped_image"]
    
    def preprocess(self):
        obj_pp = Preprocess(image=self.warped_image)
        hls_img = obj_pp.apply_colorspace(cv2.COLOR_RGB2HLS)
        h, l, s = [np.squeeze(i, axis=2) for i in np.dsplit(hls_img, 3)]
        ls_img = np.dstack([l, s, np.zeros(h.shape)]).astype(np.uint8)
        
        obj_pp.reset_image(ls_img)
        ls_gray = obj_pp.apply_colorspace(cv2.COLOR_RGB2GRAY)
        gx, gy = obj_pp.apply_gradients(kernel_size=3)
        x_abs_thresh_img = obj_pp.apply_absolute_thresh(axis="x", threshold=(15, 150))
        
        if self.save_path is not None:
            self.plot_images += [hls_img, ls_img, ls_gray, gx, gy, x_abs_thresh_img]
            self.plot_names += [
                "hls_colorspace", "ls_colorspace", "ls_gray", "gradient_x", "gradient_y", "gradient_absoute_threshold"
            ]
            
        return x_abs_thresh_img
    
    def plot(self):
        assert(len(self.plot_images) == len(self.plot_names))
        ncol = 3
        nrows = int(np.ceil(len(self.plot_names) / ncol))
        fig = commons.subplots(nrows=nrows, ncols=ncol)(self.plot_images, self.plot_names)
        commons.save_matplotlib(self.save_path, fig)


class DetectLane:
    def __init__(self):
        pass
    
