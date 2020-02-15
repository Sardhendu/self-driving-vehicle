import numpy as np
import tensorflow as tf
from src.traffic_sign_classifier.preprocess.custom_ops import tf_warp_affine, tf_shift_image


def random_zoom(min_val: float = 0.7, output_height: int=32, output_width: int=32) -> tf.Tensor:
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


def preprocess(mode: str):
    # def preprocess_(features: tf.Tensor, labels: tf.Tensor, num_classes: tf.Tensor):
    tf_random_zoom = random_zoom(min_val=0.7)
    
    def preprocess_(features, labels, num_classes):
        if mode != "predict":
            labels = tf.one_hot(labels, depth=num_classes)
            
        features = tf.cast(features, dtype=tf.float32)
        
        if mode == "train":
            def train_pp_(x):
                # features = tf.keras.preprocessing.image.random_shift(
                #         x, wrg=0.4, hrg=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest'
                # )
                # Tensorflow Augmentation Ops works on image with 0-1 range
                choice = tf.random.uniform(shape=[7], minval=0., maxval=1., dtype=tf.float32)
                
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
                        choice[3] < 0.2, lambda: x, lambda: tf.image.random_saturation(x, 0.4, 1.6)
                )
                x = tf.cond(
                        choice[4] < 0.2, lambda: x, lambda: tf_random_zoom(x)
                )

                x = tf.cond(
                        choice[5] < 0.2, lambda: x, lambda: tf_warp_affine(
                            image=x,
                            translation_xy=(np.random.randint(0, 20), np.random.randint(0, 20)),
                            rotation=np.random.randint(-20, 20), scale=1.0
                ))
                    
                x = tf.cond(
                        choice[6] < 0.2, lambda: x, lambda: tf_shift_image(
                            image=features,
                            offset_xy=(np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4))
                    )
                )

                # x = experiment(x)
                return x

            # Only Perform preprocessing
            features /= 255
            choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            features = tf.cond(choice < 0.5, lambda: features, lambda: train_pp_(features))
        else:
            features /= 255
        return tf.cast(features, dtype=tf.float32), tf.cast(labels, dtype=tf.float32) if mode != "predict" else None
    
    return preprocess_




