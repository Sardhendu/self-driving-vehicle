
import cv2
import numpy as np


class Preprocess():
    def __init__(self, image):
        self.orig_image = image
        self.processed_image = image
        self.gradient_x = None
        self.gradient_y = None
        
    def reset_image(self, image):
        self.processed_image = image
        
    def apply_blurr(self, kernel: int):
        self.processed_image = cv2.blur(self.processed_image, (kernel, kernel))
        return self.processed_image
    
    def apply_colorspace(self, color_space: cv2):
        self.processed_image = cv2.cvtColor(self.processed_image, color_space)
        return self.processed_image
    
    def apply_gradients(self, kernel_size):
        self.gradient_x = cv2.Sobel(self.processed_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        self.gradient_y = cv2.Sobel(self.processed_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        return self.gradient_x, self.gradient_y

    def apply_absolute_thresh(self, axis, threshold=(100, 150)):
        if axis == "y":
            gradient = self.gradient_y
        elif axis == "x":
            gradient = self.gradient_x
        else:
            raise ValueError('Only x and y axis permitted')
            
        print(f'absolute min={np.min(gradient)}, absolute max={np.max(gradient)}')
        abs_value = np.uint8(np.abs(gradient) / np.max(gradient) * 255)
        abs_mask = np.zeros(gradient.shape)
        abs_mask[(abs_value >= threshold[0]) & (abs_value <= threshold[1])] = 1
        return abs_mask.astype(np.int32)

    def apply_magnitude_thresh(self, threshold=(20, 150)):
        gradient_magnitude = pow(pow(self.gradient_y, 2) + pow(self.gradient_x, 2), 0.5)
        print(f'magnitude min={np.min(gradient_magnitude)}, magnitude max={np.max(gradient_magnitude)}')
        scaled_magnitude = np.uint8(np.abs(gradient_magnitude) / np.max(gradient_magnitude) * 255)
        magnitude_mask = np.zeros(self.processed_image.shape)
        magnitude_mask[(scaled_magnitude >= threshold[0]) & (scaled_magnitude <= threshold[1])] = 1
        return magnitude_mask

    def apply_orientation_thresh(self, threshold=(0.7, 1.3)):
        gradient_x, gradient_y = np.abs(self.gradient_x), np.abs(self.gradient_y)
        gradient_orientation = np.arctan2(gradient_y, gradient_x)
        print(f'orientation min={np.min(gradient_orientation)}, orientation max={np.max(gradient_orientation)}')
        orientation_mask = np.zeros(gradient_x.shape).astype(np.int32)
        orientation_mask[(gradient_orientation >= threshold[0]) & (gradient_orientation <= threshold[1])] = 1
        return orientation_mask

