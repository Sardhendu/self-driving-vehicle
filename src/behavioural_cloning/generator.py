import os
import numpy as np
import pandas as pd
import keras
from src import commons


def augmenter(driving_log_data):
    correction_val = 0.2
    center = np.array(list(driving_log_data["center"]))
    left = np.array(list(driving_log_data["left"]))
    right = np.array(list(driving_log_data["right"]))
    center_steering = np.array(driving_log_data["steering"], dtype=np.float32)
    
    left_steering = center_steering + correction_val
    right_steering = center_steering - correction_val
    
    image_paths = np.hstack((np.hstack((center, left)), right))
    steering_vals = np.hstack((np.hstack((center_steering, left_steering)), right_steering))
    
    assert (len(image_paths) == len(steering_vals))
    return image_paths, steering_vals


class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, steering_vals, batch_size, image_shape, mode: str):
        """
        This Class inherits the Keras sequence generator that generates the data in sequence for the keras fit_generator
        :param image_paths:
        :param steering_vals:
        :param batch_size:
        :param image_shape:
        :param mode:
        """
        super().__init__()
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.mode = mode
        self.gen_index = 0
        self.image_paths = image_paths
        self.steering_vals = steering_vals
        self.steps_per_epoch = int(len(self.image_paths) // self.batch_size)
        self.on_epoch_end()
    
    def on_epoch_end(self):
        ids = np.arange(0, len(self.image_paths))
        np.random.seed(None)
        np.random.shuffle(ids)
        self.image_paths = self.image_paths[ids]
        self.steering_vals = self.steering_vals[ids]
        
        self.image_paths = self.image_paths[0: self.steps_per_epoch * self.batch_size]
        self.steering_vals = self.steering_vals[0: self.steps_per_epoch * self.batch_size]
        
        self.image_paths = np.array(self.image_paths).reshape(-1, self.batch_size)
        self.steering_vals = np.array(self.steering_vals).reshape(-1, self.batch_size)
        
        self.gen_index = 0
        assert (self.image_paths.shape == self.steering_vals.shape)
    
    def __len__(self):
        """
        Denotes the number of batches per epoch
        This function should be implemented, not implementing it would
        produce not-implemented error
        This functions decides the number of batches per epoch, and only when all the batches are run
        function on_epoch_end is invoked
        :return:
        """
        
        #         print('[%s Generator] Number of Batches per epoch is set to be %s' % (str(self.mode), str(self.size)))
        return self.steps_per_epoch
    
    def __getitem__(self, index):
        """Generate one batch of data'
        :param index:
        :return:
        """
        # Generate indexes of the batch
        image_path = self.image_paths[index]
        steering_vals = self.steering_vals[index]
        
        assert (len(self.image_paths.shape) == len(self.steering_vals.shape))
        inputs, targets = self.generate(image_path, steering_vals)
        return inputs, targets
    
    def generate(self, image_paths, steering_vals):
        images = np.zeros(([self.batch_size] + self.image_shape))
        labels = np.zeros(([self.batch_size] + [1]))
        for in_, (img_path, st_val) in enumerate(zip(image_paths, steering_vals)):
            img_path = os.path.join('./data', img_path.strip())
            image = np.float32(commons.read_image(img_path))
            if self.mode != "debug":
                image, st_val, = self.preprocess(image, st_val, self.mode)
            images[in_] = image
            labels[in_] = st_val
        return images, labels
    
    def preprocess(self, image, st_val, mode):
        if mode == "train":
            if np.random.uniform() > 0.5:
                image = image[::, ::-1]
                st_val *= -1
        
        return image, st_val


def keras_generator(params, driving_log_path_list):
    for num, path in enumerate(driving_log_path_list):
        if num == 0:
            driving_log_data = pd.read_csv(path)
        else:
            driving_log_data = pd.concat([driving_log_data, pd.read_csv(path)], axis=0)

    output_shape = [160, 320, 3]

    image_paths, steering_vals = augmenter(driving_log_data)
    print("Total Count of Image paths after augmenting: ", len(image_paths))
    idx = np.arange(len(image_paths))
    np.random.seed(923)
    np.random.shuffle(idx)

    eval_prop = len(image_paths) * 0.10
    test_prop = len(image_paths) * 0.15
    
    # Fetch Test Data
    test_images = image_paths[0: int(test_prop)]
    test_steering_val = steering_vals[0: int(test_prop)]
    
    # Fetch Eval Dataset
    eval_images = image_paths[int(test_prop): int(eval_prop)+int(test_prop)]
    eval_steering_val = steering_vals[int(test_prop): int(eval_prop)+int(test_prop)]

    # Fetch Train DataSet
    train_images = image_paths[int(eval_prop)+int(test_prop):]
    train_steering_val = steering_vals[int(eval_prop)+int(test_prop):]

    train_generator = DataGenerator(
            train_images, train_steering_val, batch_size=params["batch_size"], image_shape=output_shape, mode="train"
    )
    eval_generator = DataGenerator(
            eval_images, eval_steering_val, batch_size=params["batch_size"], image_shape=output_shape, mode="eval"
    )
    test_generator = DataGenerator(
            test_images, test_steering_val, batch_size=params["batch_size"], image_shape=output_shape, mode="test"
    )

    print("[Data Count]"
          "\n\t train_images={}, train_steering_vals={}"
          "\n\t eval_images={}, eval_steering_vals={}"
          "\n\t test_images={}, test_steering_vals={}"
          "\n\t train_steps_per_epoch={}"
          "\n\t eval_steps_per_epoch={}".format(
            len(train_images),
            len(train_steering_val),
            len(eval_images),
            len(eval_steering_val),
            len(test_images),
            len(test_steering_val),
            len(train_generator),
            len(eval_generator)
    ))

    return params, train_generator, eval_generator, test_generator


if __name__ == "__main__":
    from src.behavioural_cloning.params import params
    
    driving_log_path = ["./data/driving_log.csv", "./data/driving_log2.csv"]
    params, train_generator, eval_generator, test_generator = keras_generator(
            params,
            driving_log_path_list=driving_log_path,
    )
    
    for i in range(0, len(eval_generator)):
        out = eval_generator[i]
        # print(out)
    # params, train_generator, eval_generator =
    
    # cnt_num = 0
    # for image_batch, label_batch in train_generator:
    #     print('=-=-=-=-=-=-=-=: ', cnt_num)
    #     # images, steering_val = s
    #     # print("s: ", image_batch.shape)
    #     # print('label_batch: ', label_batch)
    #     print(image_batch.shape)
    #     # print(steering_val)
    #
    #     cnt_num += 1

