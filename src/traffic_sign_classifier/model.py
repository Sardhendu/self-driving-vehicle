import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers
from typing import Callable, Dict

from src import  commons
strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

train_data_path = "./data/train.p"
valid_data_path = "./data/valid.p"

params = {
    "num_classes": 43,
    "batch_size": 256,
    "poly_decay_schedular": {
        "learning_rate": 0.01,
        "learning_power": 0.9,
        "learning_rate_min": 0.00001,
        "end_learning_rate": 0
    },
    "optimizer_learning_momentum": 0.9,
    "epochs": 10,
    "train_steps": 34799*10*256,
    "save_checkpoint": 1000,
    "save_summary_steps": 100,
    "model_dir": "./data/model"
}


def preprocess(features: tf.Tensor, labels: tf.Tensor, num_classes: tf.Tensor):
    features /= 255
    labels = tf.one_hot(labels, depth=num_classes[0])
    return tf.cast(features, dtype=tf.float32), tf.cast(labels, dtype=tf.float32)


def dataset_pipeline(images, labels, input_fn, params, mode="train"):
    num_classes = tf.constant(
            np.repeat(params["num_classes"], len(images)).astype(np.int32), dtype=tf.int32
    )

    tf.assert_equal(len(images), len(labels), len(num_classes))
    shuffle_buffer_size = np.int32(images.shape[0])
    with tf.name_scope("input_pipeline"):
        data_pipeline = tf.data.Dataset.from_tensor_slices((images, labels, num_classes))\
            .shuffle(
                buffer_size=shuffle_buffer_size,
                reshuffle_each_iteration=True)
        
        if mode == "train":
            data_pipeline = data_pipeline.repeat(params["epochs"]).batch(params["batch_size"])
        else:
            data_pipeline = data_pipeline.batch(params["batch_size"])
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
        # print('out.shape: ', out.shape)
        out = self.pool1(out)
        # print('out.shape: ', out.shape)
        out = self.conv2(out)
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
  

def loss():
    def loss_(y_true: tf.Tensor, y_logits: tf.Tensor):
        # print(f"y_true -> {y_true.shape}, y_logits -> {y_logits.shape} ")
        loss_val = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true,
            logits=y_logits,
            name=None
        )
        loss_val = tf.reduce_mean(loss_val)
        return loss_val
    return loss_
    
    
def grad(images: tf.Tensor, target_one_hot: tf.Tensor, model_builder: Callable, loss_fn: Callable):
    with tf.GradientTape() as tape:
        pred_logits = model_builder(images)
        loss_val = loss_fn(target_one_hot, pred_logits)
        
        train_vars = model_builder.trainable_variables
        
        # TODO: Try weights decay for all the weights
        gradients = tape.gradient(loss_val, train_vars)
        
        # TODO: Try gradient Norm
        return loss_val, zip(gradients, train_vars)
        
     
class SummaryCallback:
    def __init__(self, mode):
        if mode == "train":
            self.summary_writer = tf.summary.create_file_writer(f"{params['model_dir']}/")
        else:
            self.summary_writer = tf.summary.create_file_writer(f"{params['model_dir']}/eval")
        
    def scalar(self, name, value, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, tf.cast(step, tf.int64))
            self.summary_writer.flush()

        
def train_eval(
        dataset_: Callable,
        model_builder: Callable,
        loss_fn: Callable,
        learning_rate_schedular: Callable,
        train_summary_callback: Callable,
        params: Dict
):
    
    optimizer_ = tf.keras.optimizers.SGD(
            learning_rate=learning_rate_schedular, momentum=params["optimizer_learning_momentum"]
    )
    
    def step_fn(feature, target):
        """
        :param feature: [batch_size, h, w, 3]
        :param target:  [batch_size, num_classes]
        :return:
            loss_vals = [batch_size,]
        """
        loss_val, grad_vars = grad(feature, target, model_builder, loss_fn)
        optimizer_.apply_gradients(grad_vars)
        return loss_val
    
    global_step = optimizer_.iterations.numpy()
    dataset_iterator = iter(dataset_)
    progbar = tf.keras.utils.Progbar(params["train_steps"])
    loss_vals = []
    start = time.time()
    while global_step < params["train_steps"]:
        feature, target = next(dataset_iterator)
        loss_vals += [strategy.experimental_run_v2(step_fn, args=(feature, target))]
        
        if ((global_step + 1) % params["save_summary_steps"]) == 0:
            end = time.time()
            train_summary_callback.scalar("loss", tf.math.reduce_mean(loss_vals), global_step)
            loss_vals = []
            steps_per_second = params["save_summary_steps"] / (end - start)
            train_summary_callback.scalar(
                "steps_per_second", steps_per_second, step=global_step
            )
        global_step += 1
        progbar.update(global_step)
        

        
        


train_data = commons.read_pickle(train_data_path)
print(train_data.keys())

features = train_data["features"]
labels = train_data["labels"]
print(labels)

print(features.shape)
print(len(np.unique(labels)))


# output = model(tf.constant(np.random.random((1, 32, 32, 3)), dtype=tf.float32))
# print(output.shape)
from src.traffic_sign_classifier import utils
dataset_ = dataset_pipeline(features, labels, preprocess, params)
model_fn = LeNet(num_classes=43)
loss_fn = loss()
lr_schedular_fn = utils.PolyDecaySchedular(
        lr=params["poly_decay_schedular"]["learning_rate"],
        total_steps=params["train_steps"],
        learning_power=params["poly_decay_schedular"]["learning_power"],
        min_lr=params["poly_decay_schedular"]["learning_rate_min"],
        end_learning_rate=params["poly_decay_schedular"]["end_learning_rate"]
)
train_summary_callback = SummaryCallback(mode="train")
train_eval(dataset_, model_fn, loss_fn, lr_schedular_fn, train_summary_callback, params)
