import os
import numpy as np
import pandas as pd
import tensorflow as tf

from src import commons


class DataGenerator:
    def __init__(self, driving_log_path):
        driving_log_data = pd.read_csv(driving_log_path)
        correction_val = 0.2
        center = driving_log_data["center"]
        left = driving_log_data["left"]
        right = driving_log_data["right"]
        center_steering = driving_log_data["steering"]
        left_steering = np.array(
                driving_log_data["steering"], dtype=np.float32
        ) + correction_val
        right_steering = np.array(
                driving_log_data["steering"], dtype=np.float32
        ) - correction_val
        
        self.image_paths = np.hstack((np.hstack((center, left)), right))
        self.steering_vals = np.hstack((
            np.hstack((center_steering, left_steering)), right_steering
        ))
        self.gen_index = 0
        self.reset()
        
    def reset(self):
        ids = np.arange(0, len(self.image_paths))
        np.random.shuffle(ids)
        self.image_paths = self.image_paths[ids]
        self.steering_vals = self.steering_vals[ids]
        self.gen_index = 0
        
        assert (len(self.image_paths) == len(self.steering_vals))
        
    def generate(self, batch_size):
        i = 0
        while i < batch_size:
            if self.gen_index+i >= len(self.image_paths):
                self.reset()
            img_path = os.path.join('./data', self.image_paths[self.gen_index+i].strip())
            image = np.float32(commons.read_image(img_path))
            image = self.preprocess(image)
            steering_val = self.steering_vals[i]
            yield image, steering_val
            i += 1
            
        self.gen_index += batch_size
        
    def preprocess(self, image):
        image = image[40:, 0:, :]
        image /= 255
        image -= 0.5
        return image
    
    def unnormalize(self, image):
        image += 0.5
        image *= 255
        return np.uint8(image)


# driving_log_path = "./data/driving_log.csv"
# gen_ = DataGenerator(driving_log_path)
# ds_series = tf.data.Dataset.from_generator(
#         gen_.generate,
#         args=[10],
#         output_types=(tf.float32, tf.float32),
#         output_shapes=((160, 320, 3), ())
# )
#
# # ds_series_batch = ds_series.shuffle(20).padded_batch(10)
# # image_batch, label_batch = next(iter(ds_series_batch))
# # print(image_batch.numpy())
# # print()
# # print(label_batch.numpy())
#
#
# # ds_series_batch = ds_series.shuffle(20).padded_batch(10)
# cnt_num = 0
# for image_batch, label_batch in ds_series.repeat().batch(10).take(20):
#     print('=-=-=-=-=-=-=-=: ', cnt_num)
#     # images, steering_val = s
#     print("s: ", image_batch.shape)
#     print('label_batch: ', label_batch)
#     # print(images.shape)
#     # print(steering_val)
#
#     cnt_num += 1

