import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def dataset_pipeline(images, labels, input_fn, params, mode="train"):
    num_classes = tf.constant(
            np.repeat(params["num_classes"], len(images)).astype(np.int32), dtype=tf.int32
    )
    tf.assert_equal(tf.shape(images)[0], tf.shape(labels)[0], tf.shape(num_classes)[0])
    shuffle_buffer_size = np.int32(images.shape[0])
    with tf.name_scope("input_pipeline"):
        data_pipeline = tf.data.Dataset.from_tensor_slices((images, labels, num_classes))
        
        # We should map the data set first before making batches
        data_pipeline = data_pipeline.map(input_fn, num_parallel_calls=2)
        if mode == "train":
            data_pipeline = data_pipeline.repeat(params["epochs"]) \
                .shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True) \
                .batch(params["batch_size"])
        else:
            data_pipeline = data_pipeline.batch(params["batch_size"])
        return data_pipeline


class LeNet(tf.Module):
    def __init__(self, num_classes):
        self.conv1 = layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.conv2 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(120, activation="relu")
        self.dense2 = layers.Dense(84, activation="relu")
        self.logits = layers.Dense(num_classes)
    
    def __call__(self, features):
        out = self.conv1(features)
        # out = self.bn1(out)
        # print('out.shape: ', out.shape)
        out = self.pool1(out)
        # print('out.shape: ', out.shape)
        out = self.conv2(out)
        # out = self.bn2(out)
        # print('out.shape: ', out.shape)
        out = self.pool2(out)
        # print('out.shape: ', out.shape)
        out = self.flatten(out)
        # print('out.shape: ', out.shape)
        out = self.dense(out)
        # print('out.shape: ', out.shape)
        out = self.logits(out)
        # print("logits: ", out.shape)
        return out
