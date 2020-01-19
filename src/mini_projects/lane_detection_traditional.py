
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
from src.tools import subplots


def plot(image, processed_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(processed_img, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def fetch_gradients(image, kernel_size):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    return sobel_x, sobel_y


def apply_gradient_absolute_thresh(gradient, abs_thresh=(50, 150)):
    print(f'absolute min={np.min(gradient)}, absolute max={np.max(gradient)}')
    abs_value = np.uint8(np.abs(gradient)/np.max(gradient)*255)
    abs_mask = np.zeros(gradient.shape)
    abs_mask[(abs_value >= abs_thresh[0]) & (abs_value <= abs_thresh[1])] = 1
    return abs_mask

    
def apply_magnitude_thresh(gradient_x, gradient_y, magnitude_thresh=(20, 150)):
    gradient_magnitude = pow(pow(gradient_y, 2) + pow(gradient_x, 2), 0.5)
    print(f'magnitude min={np.min(gradient_magnitude)}, magnitude max={np.max(gradient_magnitude)}')
    scaled_magnitude = np.uint8(np.abs(gradient_magnitude) / np.max(gradient_magnitude) * 255)
    magnitude_mask = np.zeros(gradient_x.shape)
    magnitude_mask[(scaled_magnitude >= magnitude_thresh[0]) & (scaled_magnitude <= magnitude_thresh[1])] = 1
    return magnitude_mask
 
 
def apply_orientation_thresh(gradient_x, gradient_y, orientation_thresh=(0.7, 1.3)):
    gradient_x, gradient_y = np.abs(gradient_x), np.abs(gradient_y)
    gradient_orientation = np.arctan2(gradient_y, gradient_x)
    print(f'orientation min={np.min(gradient_orientation)}, orientation max={np.max(gradient_orientation)}')
    orientation_mask = np.zeros(gradient_x.shape).astype(np.int32)
    orientation_mask[(gradient_orientation >= orientation_thresh[0]) & (gradient_orientation <= orientation_thresh[1])] = 1
    return orientation_mask
    

def detect_lane_rgb_gradient(
        kernel_size, magnitude_thresh=(0, 255), orientation_thresh=(0, np.pi/2), abs_thresh=(0,255)
):
    image_ = imageio.imread("data/signs_vehicles_xygrad.jpg")
    image = image_.copy()
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    sobel_x, sobel_y = fetch_gradients(image, kernel_size)
    abs_mask_img = apply_gradient_absolute_thresh(sobel_x, abs_thresh=(20, 150))
    mag_mask_img = apply_magnitude_thresh(sobel_x, sobel_y, magnitude_thresh=(50, 150))
    orient_mask_img = apply_orientation_thresh(sobel_x, sobel_y, orientation_thresh=(0.7, 1.3))
    
    final_mask = np.zeros(image_.shape)
    final_mask[(abs_mask_img==1) & (mag_mask_img==1) & (orient_mask_img==1)] = 1
    
    # plot(image, abs_mask_img)
    plot(image, final_mask)
    
    
def detect_lane_hls_gradient(
        kernel_size, gradient_channel="s"
):
    image = imageio.imread("data/bridge_shadow.jpg")
    hls_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    print(f'[Shape] hls={hls_img.shape}, gray_img={gray_img.shape}')
    
    h, l, s = [np.squeeze(i, axis=2) for i in np.dsplit(hls_img, 3)]
    print(f'[Shape] h={h.shape}, l={l.shape}, s={s.shape}')

    sobel_gx, sobel_gy = fetch_gradients(gray_img, kernel_size=3)
    gray_gradient_thresh = apply_magnitude_thresh(gradient_x=sobel_gx, gradient_y=sobel_gy, magnitude_thresh=(50, 150))

    sobel_hx, sobel_hy = fetch_gradients(h, kernel_size=3)
    h_gradient_mag_thresh = apply_magnitude_thresh(gradient_x=sobel_hx, gradient_y=sobel_hy, magnitude_thresh=(20, 150))
    h_gradient_thresh = apply_gradient_absolute_thresh(gradient=sobel_hx, abs_thresh=(10, 150))
    
    sobel_lx, sobel_ly = fetch_gradients(l, kernel_size=3)
    l_gradient_mag_thresh = apply_magnitude_thresh(gradient_x=sobel_lx, gradient_y=sobel_ly, magnitude_thresh=(20, 150))
    l_gradient_thresh = apply_gradient_absolute_thresh(gradient=sobel_lx, abs_thresh=(10, 150))
    
    sobel_sx, sobel_sy = fetch_gradients(s, kernel_size=3)
    s_gradient_mag_thresh = apply_magnitude_thresh(gradient_x=sobel_sx, gradient_y=sobel_sy, magnitude_thresh=(20, 150))
    s_gradient_thresh = apply_gradient_absolute_thresh(gradient=sobel_sx, abs_thresh=(10, 150))
    # s_gradient_oreint_thresh = apply_orientation_thresh(sobel_sx, sobel_sy, orientation_thresh=(0.7, 1.3))

    # It seems both the channels (l and s) are better in finding different part of lane. Using an AND operation may
    # not work. using an OR operation could bring
    zeros = np.zeros_like(s_gradient_thresh)
    ot = np.dstack([zeros, l_gradient_thresh, s_gradient_thresh]) * 255
    
    

    
    
    # ls_thresh = np.zeros(s.shape).astype(np.int32)
    # ls_thresh[]
    plts = subplots(nrows=4, ncols=3, figsize=(50, 20))
    plts(
            [image, gray_img, hls_img,
             h, l, s,
             h_gradient_thresh, l_gradient_thresh, s_gradient_thresh, ot],
            ["raw_img", "gray_img", "hls_img",
             "h_channel", "l_channel", "s_channel",
             "h_gradient_thresh", "l_gradient_thresh", "s_gradient_thresh", "stacked_output"]
    )
    plt.show()
    
    
    
    
detect_lane_hls_gradient(kernel_size=3)