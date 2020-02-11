
import tensorflow as tf
from typing import Tuple


def random_zoom(min_val: float = 0.8, output_height: int=32, output_width: int=32) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.linspace(min_val, 1, 20))
    boxes = np.zeros((len(scales), 4))
    box_indices = np.zeros(len(scales))
    
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]
    
    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize(
                [img], boxes=boxes, box_indices=box_indices, crop_size=(output_height, output_width)
        )
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
    return random_crop


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


def preprocess(mode: str):
    # def preprocess_(features: tf.Tensor, labels: tf.Tensor, num_classes: tf.Tensor):
    tf_random_zoom = random_zoom(min_val=0.8)
    
    def preprocess_(features, labels, num_classes):
        labels = tf.one_hot(labels, depth=num_classes)
        features = tf.cast(features, dtype=tf.float32)
        
        if mode == "train":
            def train_pp_(x):
                # Tensorflow Augmentation Ops works on image with 0-1 range
                choice = tf.random.uniform(shape=[5], minval=0., maxval=1., dtype=tf.float32)
                x = tf.cond(
                        choice[0] < 0.2, lambda: x, lambda: tf.image.random_brightness(x, max_delta=0.4)
                )
                x = tf.cond(
                        choice[1] < 0.2, lambda: x, lambda: tf.image.random_contrast(x, 0.6, 1.6)
                )
                # x = tf.cond(
                #         choice[2] < 0.5, lambda: x, lambda: tf.image.random_hue(x, 0.03)
                # )
                x = tf.cond(
                        choice[3] < 0.2, lambda: x, lambda: tf.image.random_saturation(x, 0.5, 1.5)
                )
                x = tf.cond(
                        choice[4] < 0.2, lambda: x, lambda: tf_random_zoom(x)
                )
                return x

            # Only Perform preprocessing
            features /= 255
            choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            features = tf.cond(choice < 0.5, lambda: features, lambda: train_pp_(features))
        else:
            features /= 255
        return tf.cast(features, dtype=tf.float32), tf.cast(labels, dtype=tf.float32)
    
    return preprocess_


def preprocess_test(features, method):
    # labels = tf.one_hot(labels, depth=num_classes[0])
    
    if method == "random_shift":
        features = tf.keras.preprocessing.image.random_shift(
                features, wrg=0.4, hrg=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
        )
    if method == "random_zoom":
        features /= 255
        features = random_zoom(features)
        features *= 255
    if method == "random_brightness":
        # Tensorflow opeation works on 0-1 range
        features /= 255
        # features = tf.keras.preprocessing.image.random_brightness(features, brightness_range=(0.6, 1.4))
        features = tf.image.random_brightness(features, max_delta=0.2)
        features *= 255
    
    if method == "random_contrast":
        features /= 255
        features = tf.image.random_contrast(features, 0.6, 1.6)
        features *= 255
    
    if method == "random_hue":
        features /= 255
        features = tf.image.random_hue(features, 0.03)
        features *= 255
    
    if method == "random_saturation":
        features /= 255
        features = tf.image.random_saturation(features, 0.5, 1.5)
        features *= 255
    
    if method == "random_flip":
        features = tf.image.flip_left_right(features)
    
    return tf.cast(features, dtype=tf.uint8)


if __name__ == "__main__":
    import os
    import numpy as np
    from src import commons
    
    np.random.seed(365)
    
    method = "random_saturation"
    train_data_path = "./data/train.p"
    valid_data_path = "./data/valid.p"
    train_data = commons.read_pickle(train_data_path)
    eval_data = commons.read_pickle(valid_data_path)
    
    train_features = train_data["features"]
    train_labels = train_data["labels"]
    eval_features = eval_data["features"]
    eval_labels = eval_data["labels"]
    print(f"[Train]: features={train_features.shape}, labels={train_labels.shape}")
    print(f"[Train]: features={eval_features.shape}, labels={eval_labels.shape}")
    
    im_size = 32
    count_of_images_in_x_dir = 5
    count_of_images_in_y_dir = 4
    inner_padding = 5
    output_padding = 10
    
    one_img = im_size + inner_padding + im_size
    x_pxls_cnt = count_of_images_in_x_dir * (one_img) + output_padding * (count_of_images_in_x_dir - 1)
    y_pxls_cnt = count_of_images_in_y_dir * im_size + output_padding * (count_of_images_in_y_dir - 1)
    
    output_image = np.zeros((y_pxls_cnt, x_pxls_cnt, 3))
    
    to_y = 0
    to_x = 0
    running_cnt = 0
    for y in range(0, count_of_images_in_y_dir):
        from_y = to_y + output_padding if y > 0 else 0
        to_y = from_y + im_size
        print('')
        for x in range(0, count_of_images_in_x_dir):
            from_x = to_x + output_padding if x > 0 else 0
            to_x = from_x + (im_size + inner_padding + im_size)
            
            idx = np.random.randint(0, len(train_features))
            feature = train_features[idx]
            feature_in = tf.constant(feature, dtype=tf.float32)
            print(type(feature_in))
            out = preprocess_test(feature_in, method=method)
            preprocessed_data = out.numpy()
            
            out_data = np.column_stack(
                    (feature, np.zeros((im_size, inner_padding, 3)), preprocessed_data)
            ).astype(np.uint8)
            
            output_image[from_y:to_y, from_x:to_x, :] = out_data
            running_cnt += 1
    
    os.makedirs("./data/preprocessed_img", exist_ok=True)
    commons.save_image(f'./data/preprocessed_img/{method}.png', output_image)
    
    # print('')
    # for num, feature in enumerate(train_features):
    #     print(from_, to_)
    #     from_ = to_
    #     # print(feature.shape)
    #     # out = preprocess_test(feature)
    #     # preprocessed_data = out.numpy()
    #     # out_data = np.column_stack(
    #     #     (feature, np.zeros((32, 10, 3)), preprocessed_data)
    #     # ).astype(np.uint8)
    #     # os.makedirs("./data/preprocessed_img/random_shift", exist_ok=True)
    #     # commons.save_image(f'./data/preprocessed_img/random_shift/{num}.png', out_data)
    #     if num == 10:
    #         break



