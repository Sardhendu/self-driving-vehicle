import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src import  commons

train_data_path = "./data/train.p"
valid_data_path = "./data/valid.p"


def preprocess(features, labels):
    return features, labels

def dataset_pipeline(images, labels, input_fn, epochs):
    shuffle_buffer_size = np.int32(images.shape[0])
    with tf.name_scope("input_pipeline"):
        data_pipeline = tf.data.Dataset.from_tensor_slices((images, labels))\
            .shuffle(
                buffer_size=shuffle_buffer_size,
                reshuffle_each_iteration=True)
        
        data_pipeline = data_pipeline.repeat(epochs).batch(10)
        data_pipeline = data_pipeline.map(input_fn, num_parallel_calls=2)
        
        return data_pipeline


class LeNet(tf.Module):
    def __init__(self, num_classes):
        self.conv1 = layers.Conv2D(filters=6, kernel_size=(5, 5), strides=1, activation="relu")
        self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=1, activation="relu")
        self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(120, activation="relu")
        self.dense2 = layers.Dense(84, activation="relu")
        self.logits = layers.Dense(num_classes)
        
    def __call__(self, features):
        out = self.conv1(features)
        print('out.shape: ', out.shape)
        out = self.pool1(out)
        print('out.shape: ', out.shape)
        out = self.conv2(out)
        print('out.shape: ', out.shape)
        out = self.pool2(out)
        print('out.shape: ', out.shape)
        out = self.flatten(out)
        print('out.shape: ', out.shape)
        out = self.dense(out)
        print('out.shape: ', out.shape)
        out = self.logits(out)
        print("logits: ", out.shape)
        return out
    

train_data = commons.read_pickle(train_data_path)
print(train_data.keys())

features = train_data["features"]
labels = train_data["labels"]
print(labels)

print(features.shape)
print(len(np.unique(labels)))

model = LeNet(num_classes=43)
output = model(tf.constant(np.random.random((1,32,32,3)), dtype=tf.float32))
print(output.shape)

dataset_ = dataset_pipeline(features, labels, preprocess, epochs=2)
dataset_iterator = iter(dataset_)
for i in range(0, 100):
    feature, label = next(dataset_iterator)
    print(feature.numpy().shape, label.numpy())
    
    