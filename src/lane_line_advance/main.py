
import cv2
import numpy as np
from shapely import geometry as geom

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
        ones = np.ones(cnt_left_lane_pnts+cnt_right_lane_pnts).reshape(-1, 1)
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
        if self.save_path:
            unwarped_image = cv2.warpPerspective(
                    self.obj_img_plots.image.copy(), self.M, (self.w, self.h), flags=cv2.INTER_NEAREST
            )
            self.obj_img_plots.polymask(
                    points=np.vstack((left_lane, right_lane[::-1])),
                    color = (0, 255, 0),
                    mask_weight=0.5
            )
            self.plot_images += [unwarped_image, self.obj_img_plots.image]
            self.plot_names += ["unwarped_image", "transformed_lane_points"]
        return transformed_points
    
    def plot(self):
        assert(len(self.plot_images) == len(self.plot_names))
        ncol = 3
        nrows = int(np.ceil(len(self.plot_names) / ncol))
        fig = commons.subplots(nrows=nrows, ncols=ncol)(self.plot_images, self.plot_names)
        commons.save_matplotlib(self.save_path, fig)
            

class DetectLane:
    def __init__(self):
        pass

# from src import commons
# from src.lane_line_advance.main import PreprocessingPipeline
# import matplotlib.pyplot as plt
# test_image_name = "test4"#"straight_lines1"  # test4
#
# input_image_path = f'./data/test_images/{test_image_name}.jpg'
# output_plot_path = f'./data/output_images/{test_image_name}.png'
# hist_output_path = f"./data/output_images/hist_{test_image_name}.png"
# curvature_bbox_output_path = f"./data/output_images/curvature_{test_image_name}.png"
#
# image = commons.read_image(input_image_path)
# print(image.shape) # (720, 1280, 3)
#
# def unwarp(dst_points=[(500, 450), (0, 700), (750, 450), (1250, 700)]):
#     h, w, _ = image.shape
#     src_points = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)] # [(0, 0), (0, 719), (1279, 0), (1279, 719)]
#     print('src_points: ', src_points)
#     M = cv2.getPerspectiveTransform(
#             np.array(src_points).astype(np.float32), np.array(dst_points).astype(np.float32)
#     )
#     # cv2.wa
#     print(M)
#
#     for
#     out = np.dot(M, np.array([0, 719, 1]))
#     print(out/out[2])
#     warped_image = cv2.warpPerspective(image.copy(), M, (w, h), flags=cv2.INTER_NEAREST)
#     _ = commons.subplots()([warped_image])
#     # plt.show()
#     # if save_path:
#     #     plot_images += [self.warped_image]
#     #     self.plot_names += ["warped_image"]
#
#
# unwarp()




"""

[ 1.95465207e-01 -6.95410292e-01  5.00000000e+02]
 [ 7.99985566e-17 -4.31154381e-01  4.50000000e+02]
 [ 1.77774570e-19 -1.11265647e-03  1.00000000e+00]]
 
 
[[-8.25161290e-01 -1.65032258e+00  1.15522581e+03]
 [ 0.00000000e+00 -2.31935484e+00  1.04370968e+03]
 [ 1.77774570e-19 -2.58064516e-03  1.00000000e+00]]


[[-1.21188428e+00  4.31154381e+00 -3.10000000e+03]
 [-6.01083760e-16  2.67315716e+00 -2.79000000e+03]
 [-1.33574169e-18  6.89847010e-03 -6.20000000e+00]]
"""