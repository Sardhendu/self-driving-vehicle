import tensorflow as tf
from typing import Tuple


def tf_warp_affine(
        image: tf.Tensor,
        translation_xy: Tuple[float, float],
        rotation: float,
        scale: float
):
    pi_on_180 = 0.017453292519943295
    angle = rotation * pi_on_180
    
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
    
    T = tf.convert_to_tensor([
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
        x_flat = tf.repeat(tf.range(tf.shape(image)[1]), repeats=tf.shape(image)[0])
        y_flat = tf.tile(tf.range(tf.shape(image)[0]), [tf.shape(image)[1]])
        
        output_indices = tf.stack([x_flat, y_flat, tf.ones(tf.shape(y_flat)[0], dtype=tf.int32)], axis=0)
        return x_flat, y_flat, output_indices
    
    # Step 1: Create the Transformation matrix
    A_tf = tf.cast(tf.matmul(tf.matmul(tf.matmul(T, R), S), tf.linalg.inv(T)), dtype=tf.float32)
    
    # Step 2: Extract x and y coordinates form the image
    x_orig_coord, y_orig_coords, homogenous_coords_tf = get_grid_tf()
    
    # Step 3: Project the coordinates into the transformed space
    warped_coords = tf.round(tf.matmul(A_tf, tf.cast(homogenous_coords_tf, dtype=tf.float32)))
    x_warped_coords = warped_coords[0]
    y_warped_coord = warped_coords[1]
    
    # Step 4: Capture only the indices that fall within the original image scope
    height = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    width = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    
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
    pxl_vals = tf.cast(tf.gather_nd(image, tf.stack([y_orig_bound, x_orig_bound], axis=1)), tf.float32)
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
    return transformed_image


def tf_shift_image(image: tf.Tensor, offset_xy: Tuple[float, float]):
    """
    :param image:           (h, w, c) The input image to shift
    :param offset_xy:       (-1, 1) offset to shift in x and y dir
    :return:
    """
    offset_x, offset_y = offset_xy
    height_flt = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    width_flt = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    height_int = tf.cast(tf.shape(image)[0], dtype=tf.int32)
    width_int = tf.cast(tf.shape(image)[1], dtype=tf.int32)
    shift_x_val = tf.cast(tf.round(tf.multiply(offset_x, width_flt)), dtype=tf.int32)
    shift_y_val = tf.cast(tf.round(tf.multiply(offset_y, height_flt)), dtype=tf.int32)
    
    y_flat = tf.cast(tf.repeat(tf.range(height_int), repeats=width_int), dtype=tf.int32)
    x_flat = tf.cast(tf.tile(tf.range(width_int), [height_int]), dtype=tf.int32)
    
    y_shift = y_flat + shift_y_val
    x_shift = x_flat + shift_x_val
    
    indices = tf.where(
            (x_shift >= 0) &
            (x_shift < width_int) &
            (y_shift >= 0) &
            (y_shift < height_int)
    )
    # print('y_shift: ', y_shift)
    # print('x_shift: ', x_shift)
    # print('indices: ', indices)
    
    y_shift_active = tf.gather_nd(y_shift, indices)
    x_shift_active = tf.gather_nd(x_shift, indices)
    y_orig_active = tf.gather_nd(y_flat, indices)
    x_orig_active = tf.gather_nd(x_flat, indices)
    
    pxl_vals = tf.cast(tf.gather_nd(image, tf.stack([y_orig_active, x_orig_active], axis=1)), tf.float32)
    # print("pxl_vals: ", pxl_vals)
    
    # obj_lookup = LookUp(y_orig_active, x_orig_active, pxl_vals)
    
    ch1_st = tf.SparseTensor(
            indices=tf.cast(tf.stack([y_shift_active, x_shift_active], axis=1), dtype=tf.int64),
            values=tf.cast(pxl_vals[:, 0], dtype=tf.float32),
            dense_shape=[height_int, width_int]
    )
    ch1 = tf.sparse.to_dense(ch1_st, default_value=0, validate_indices=False)
    
    ch2_st = tf.SparseTensor(
            indices=tf.cast(tf.stack([y_shift_active, x_shift_active], axis=1), dtype=tf.int64),
            values=tf.cast(pxl_vals[:, 1], dtype=tf.float32),
            dense_shape=[height_int, width_int]
    )
    ch2 = tf.sparse.to_dense(ch2_st, default_value=0, validate_indices=False)
    
    ch3_st = tf.SparseTensor(
            indices=tf.cast(tf.stack([y_shift_active, x_shift_active], axis=1), dtype=tf.int64),
            values=tf.cast(pxl_vals[:, 2], dtype=tf.float32),
            dense_shape=[height_int, width_int]
    )
    ch3 = tf.sparse.to_dense(ch3_st, default_value=0, validate_indices=False)
    
    transformed_image = tf.stack([ch1, ch2, ch3], axis=-1)
    
    # obj_lookup.fill(transformed_image)
    
    return transformed_image
