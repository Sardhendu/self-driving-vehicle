import numpy as np
from typing import Tuple
import tensorflow as tf
import imageio
import numpy as np
import cv2


def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    print(coords.shape)
    print(coords)
    coords = np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int32) if homogenous else coords.astype(np.int32)
    print(coords)
    return coords


def affine_transform(
        image: tf.Tensor, translation_xy: Tuple[float, float], rotation: float, scale: float
):
    pi_on_180 = 0.017453292519943295
    angle = rotation * pi_on_180
    height, width, _ = tf.shape(image)
    
    # Construct the Affine Matrices
    R = tf.convert_to_tensor([
        [
            tf.math.cos(angle),
            tf.math.sin(angle),
            tf.convert_to_tensor(0, dtype=tf.float32)
        ],
        [
            -tf.math.sin(angle),
            tf.math.cos(angle),
            tf.convert_to_tensor(0, dtype=tf.float32)
        ],
        [
            tf.convert_to_tensor(0, dtype=tf.float32),
            tf.convert_to_tensor(0, dtype=tf.float32),
            tf.convert_to_tensor(1, dtype=tf.float32)]
    ], dtype=tf.float32
    )
    
    T = tf.constant([
        [1, 0, translation_xy[0]],
        [0, 1, translation_xy[1]],
        [0, 0, 1]
    ], dtype=tf.float32)
    
    S = tf.constant([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype=tf.float32)
    
    def get_grid_tf():
        x_flat = tf.repeat(tf.range(width), repeats=height)
        y_flat = tf.tile(tf.range(height), [width])
        
        output_indices = tf.stack([x_flat, y_flat, tf.ones(tf.shape(y_flat)[0], dtype=tf.int32)], axis=0)
        return x_flat, y_flat, output_indices
    
    # Step 1: Create the Transformation matrix
    A_tf = tf.cast(tf.matmul(tf.matmul(tf.matmul(T, R), S), tf.linalg.inv(T)), dtype=tf.float32)
    
    # Step 2: Extract x and y coordinates form the image
    x_orig_coord, y_orig_coords, homogenous_coords_tf = get_grid_tf()
  
    # Step 3: Project the coordinates into the transformed space
    warped_coords = tf.round(tf.matmul(A_tf, tf.cast(homogenous_coords_tf, dtype=tf.float32)))
    x_warped_coords, y_warped_coord, _ = warped_coords

    # Step 4: Capture only the indices that fall within the original image scope
    height = tf.cast(height, dtype=tf.float32)
    width = tf.cast(width, dtype=tf.float32)
    
    indices = tf.where(
            (x_warped_coords >= 0.0) &
            (x_warped_coords < width) &
            (y_warped_coord >= 0.0) &
            (y_warped_coord < height)
    )

    # Step 5: Gather , Indices from transformed image
    x_warp_bound = tf.cast(tf.gather_nd(x_orig_coord, indices), dtype=tf.int32)
    y_warp_bound = tf.cast(tf.gather_nd(y_warped_coord, indices), dtype=tf.int32)
    x_orig_bound = tf.gather_nd(x_orig_coord, indices)
    y_orig_bound = tf.gather_nd(y_orig_coords, indices)
    
    # Step 6: (Construct the Transformed Image)Gather pxl values using Transformed indices
    pxl_vals = tf.cast(tf.gather_nd(image, tf.stack([y_orig_bound, x_orig_bound], axis=1)), tf.int32)
    ch1_st = tf.SparseTensor(
            indices=tf.cast(tf.stack([y_warp_bound, x_warp_bound], axis=1), dtype=tf.int64),
            values=tf.cast(pxl_vals[:, 0], dtype=tf.float32),
            dense_shape=[height, width]
    )
    ch1 = tf.sparse.to_dense(ch1_st, default_value=0, validate_indices=False)
    
    ch2_st = tf.SparseTensor(
            indices=tf.cast(tf.stack([y_warp_bound, x_warp_bound], axis=1), dtype=tf.int64),
            values=tf.cast(pxl_vals[:, 1], dtype=tf.float32),
            dense_shape=[height, width]
    )
    ch2 = tf.sparse.to_dense(ch2_st, default_value=0, validate_indices=False)
    
    ch3_st = tf.SparseTensor(
            indices=tf.cast(tf.stack([y_warp_bound, x_warp_bound], axis=1), dtype=tf.int64),
            values=tf.cast(pxl_vals[:, 2], dtype=tf.float32),
            dense_shape=[height, width]
    )
    ch3 = tf.sparse.to_dense(ch3_st, default_value=0, validate_indices=False)
    
    transformed_image = tf.stack([ch1, ch2, ch3], axis=-1)
    transformed_image = tf.image.resize(
            transformed_image,
            tf.shape(transformed_image)[0:2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            preserve_aspect_ratio=False,
            antialias=False,
            name=None
    )
    return transformed_image


from src import commons
train_data_path = './data/train.p'
save_path = "./data/save_images"
data = commons.read_pickle(train_data_path)
features = data["features"]
labels = data["labels"]


image = features[0]
print('image: ', image.shape)
translation_xy_val = (10.0, 2.0)
image_out = affine_transform(image, translation_xy=translation_xy_val, rotation=float(-10), scale=1.0)
aa = image_out.numpy().astype(np.uint8)

cv2.imwrite(f"{save_path}/warped_1.png", aa)
