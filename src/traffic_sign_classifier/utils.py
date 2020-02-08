import tensorflow as tf
from typing import Any


def preprocess(mode: str):
    # def preprocess_(features: tf.Tensor, labels: tf.Tensor, num_classes: tf.Tensor):

    def preprocess_(features, labels, num_classes):
        labels = tf.one_hot(labels, depth=num_classes)
        features = tf.cast(features, dtype=tf.float32)
        
        if mode == "train":
            # Tensorflow Augmentation Ops works on image with 0-1 range
            features /= 255
            features = tf.image.random_brightness(features, max_delta=0.6, seed=1)
            features = tf.image.random_contrast(features, 0.6, 1.6)
            features = tf.image.random_hue(features, 0.03)
            features = tf.image.random_saturation(features, 0.6, 1.6)
        else:
            features /= 255
        return tf.cast(features, dtype=tf.float32), tf.cast(labels, dtype=tf.float32)
    
    return preprocess_


class SummaryCallback:
    def __init__(self, model_dir, mode):
        if mode == "train":
            self.summary_writer = tf.summary.create_file_writer(f"{model_dir}/")
        else:
            self.summary_writer = tf.summary.create_file_writer(f"{model_dir}/eval")

    def scalar(self, name, value, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, tf.cast(step, tf.int64))
            self.summary_writer.flush()

        
def preprocess_test(features, method):
    
    # labels = tf.one_hot(labels, depth=num_classes[0])
    
    if method == "random_shift":
        features = tf.keras.preprocessing.image.random_shift(
                features, wrg=0.4, hrg=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
        )
    
    if method == "random_zoom":
        zoom_out = 0.6  # Magnifies the Image by 40%
        zoom_in = 1.4  # Shrinks the image by 40%
        features = tf.keras.preprocessing.image.random_zoom(
            features, (zoom_out, zoom_in), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest',
        )

    if method == "random_brightness":
        # Tensorflow opeation works on 0-1 range
        features /= 255
        # features = tf.keras.preprocessing.image.random_brightness(features, brightness_range=(0.6, 1.4))
        features = tf.image.random_brightness(features, max_delta=0.8, seed=1)
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
        features = tf.image.random_saturation(features, 0.6, 1.6)
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
    count_of_images_in_y_dir = 8
    inner_padding = 5
    output_padding = 10
    
    one_img = im_size+inner_padding+im_size
    x_pxls_cnt = count_of_images_in_x_dir*(one_img) + output_padding*(count_of_images_in_x_dir-1)
    y_pxls_cnt = count_of_images_in_y_dir*im_size + output_padding*(count_of_images_in_y_dir-1)

    output_image = np.zeros((y_pxls_cnt, x_pxls_cnt, 3))

    to_y = 0
    to_x = 0
    running_cnt = 0
    for y in range(0, count_of_images_in_y_dir):
        from_y = to_y + output_padding if y > 0 else 0
        to_y = from_y + im_size
        print('')
        # print('yyyyyy: ', from_y, to_y)
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



