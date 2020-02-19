import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from src import commons


def augmenter(driving_log_data):
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

    image_paths = np.hstack((np.hstack((center, left)), right))
    steering_vals = np.hstack((
        np.hstack((center_steering, left_steering)), right_steering
    ))
    
    assert (len(image_paths) == len(steering_vals))
    return image_paths, steering_vals


class DataGenerator:
    def __init__(self, image_paths, steering_vals, mode: bool):
        self.mode = mode
        self.gen_index = 0
        self.image_paths = image_paths
        self.steering_vals=steering_vals
        self.reset()
        
    def reset(self):
        ids = np.arange(0, len(self.image_paths))
        np.random.seed(None)
        np.random.shuffle(ids)
        self.image_paths = self.image_paths[ids]
        self.steering_vals = self.steering_vals[ids]
        assert (len(self.image_paths) == len(self.steering_vals))
        self.gen_index = 0
        
        assert (len(self.image_paths) == len(self.steering_vals))
        
    def generate(self, epochs):
        i = 0
        self.gen_index = 0
        while i < len(self.image_paths)*epochs:
            if self.gen_index >= len(self.image_paths):
                self.reset()

            img_path = os.path.join('./data', self.image_paths[self.gen_index].strip())
            print("img_path: ", i, self.gen_index, len(self.image_paths), img_path)
            image = np.float32(commons.read_image(img_path))
            if self.mode != "debug":
                image = self.preprocess(image)
            steering_val = self.steering_vals[self.gen_index]
            yield image, steering_val

            self.gen_index += 1
            i += 1
        
    def preprocess(self, image):
        image = image[40:, 0:, :]
        image /= 255
        image -= 0.5
        return image
    

def tf_generator(params, driving_log_path, mode):
    if mode == "debug":
        output_shape = (160, 320, 3)
    else:
        output_shape = (120, 320, 3)
        
    driving_log_data = pd.read_csv(driving_log_path)
    image_paths, steering_vals = augmenter(driving_log_data)
    # print(image_paths)
    idx = np.arange(len(image_paths))
    np.random.seed(923)
    np.random.shuffle(idx)

    eval_prop = len(image_paths)*0.15
    eval_images = image_paths[0: int(eval_prop)]
    eval_steering_val = steering_vals[0: int(eval_prop)]
    
    train_images = image_paths[int(eval_prop):]
    train_steering_val = steering_vals[int(eval_prop):]
    
    train_steps_per_epoch = int(len(train_images) // params["batch_size"])
    eval_steps_per_epoch = int(len(eval_images) // params["batch_size"])
    
    print(f"[Data Count]"
          f"\n\t train_images={len(train_images)}, train_steering_vals={len(train_steering_val)}"
          f"\n\t eval_images={len(eval_images)}, eval_steering_vals={len(eval_steering_val)}"
          f"\n\t train_steps_per_epoch={train_steps_per_epoch}"
          f"\n\t eval_steps_per_epoch={eval_steps_per_epoch}")
    
    train_generator_obj = DataGenerator(train_images, train_steering_val, mode=mode)
    eval_generator_obj = DataGenerator(eval_images, eval_steering_val, mode=mode)

    train_generator = tf.data.Dataset.from_generator(
            train_generator_obj.generate,
            args=[params["epochs"]],
            output_types=(tf.float32, tf.float32),
            output_shapes=(output_shape, ())
    )
    
    eval_generator = tf.data.Dataset.from_generator(
            eval_generator_obj.generate,
            args=[1],
            output_types=(tf.float32, tf.float32),
            output_shapes=(output_shape, ())
    )
    
    train_generator = train_generator.repeat()\
        .batch(params["batch_size"])\
        .take(train_steps_per_epoch*params["epochs"])
    
    eval_generator = eval_generator.repeat()\
        .batch(params["batch_size"])\
        .take(eval_steps_per_epoch)
    
    return train_generator, eval_generator


if __name__ == "__main__":
    from src.behavioural_cloning.params import params
    driving_log_path = "./data/driving_log.csv"
    train_generator, eval_generator = tf_generator(
            params,
            driving_log_path=driving_log_path,
            mode="debug"
    )
    
    cnt_num = 0
    for image_batch, label_batch in train_generator:
        print('=-=-=-=-=-=-=-=: ', cnt_num)
        # images, steering_val = s
        # print("s: ", image_batch.shape)
        # print('label_batch: ', label_batch)
        print(image_batch.shape)
        # print(steering_val)

        cnt_num += 1

