import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers
from typing import Callable, Dict, Any

from src import  commons
from src.traffic_sign_classifier.params import params
from src.traffic_sign_classifier.utils import PolyDecaySchedular, SummaryCallback, preprocess

strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")


def dataset_pipeline(images, labels, input_fn, params, mode="train"):
    num_classes = tf.constant(
            np.repeat(params["num_classes"], len(images)).astype(np.int32), dtype=tf.int32
    )

    tf.assert_equal(len(images), len(labels), len(num_classes))
    shuffle_buffer_size = np.int32(images.shape[0])
    with tf.name_scope("input_pipeline"):
        data_pipeline = tf.data.Dataset.from_tensor_slices((images, labels, num_classes, mode))\
            .shuffle(
                buffer_size=shuffle_buffer_size,
                reshuffle_each_iteration=True
        )

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


def eval(
        eval_dataset: Callable,
        model_builder: Callable,
        loss_fn: Callable,
        eval_summary_writer: SummaryCallback,
        params: Dict
):
    eval_loss = tf.keras.metrics.Sum("eval_loss", dtype=tf.float32)
    eval_acc = tf.keras.metrics.Accuracy("eval_accuracy", dtype=tf.float32)
    
    def eval_(global_step):
        iterator_ = iter(eval_dataset)
        progbar = tf.keras.utils.Progbar(params["eval_data_cnt"])
        eval_loss.reset_states()
        eval_acc.reset_states()
        it = 0
        while it < params["eval_data_cnt"]:
            eval_images, eval_target_one_hot = next(iterator_)
            eval_pred_logits = model_builder(eval_images)
            
            loss_eval = loss_fn(eval_target_one_hot, eval_pred_logits)
            eval_loss.update_state(loss_eval)
            
            labels = tf.argmax(eval_target_one_hot, axis=-1)
            preds = tf.argmax(eval_pred_logits, axis=-1)
            eval_acc.update_state(y_true=labels, y_pred=preds)
            
            it += 1
            progbar.update(it)
        avg_eval_loss = tf.divide(eval_loss.result(), params["eval_data_cnt"])
        eval_summary_writer.scalar("eval_loss", avg_eval_loss, global_step)
        eval_summary_writer.scalar("eval_accuracy", eval_acc.result(), global_step)
        
    return eval_


def train_eval(
        dataset_: Callable,
        model_builder: Callable,
        loss_fn: Callable,
        learning_rate_schedular: Callable,
        eval_callback: Callable,
        train_summary_writer: Any,
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
    progbar = tf.keras.utils.Progbar(params["train_steps"]*params["epochs"])
    train_loss = tf.keras.metrics.Sum("train_loss", dtype=tf.float32)

    start = time.time()
    while global_step < (params["train_steps"]*params["epochs"]):
        feature, target = next(dataset_iterator)
        loss_vals = strategy.experimental_run_v2(step_fn, args=(feature, target))
        train_loss.update_state([loss_vals])

        if ((global_step + 1) % params["save_summary_steps"]) == 0:
            end = time.time()
            avg_train_loss = tf.divide(train_loss.result(), params["save_summary_steps"])
            train_summary_writer.scalar("loss", avg_train_loss, global_step)
            train_loss.reset_states()
            steps_per_second = params["save_summary_steps"] / (end - start)
            train_summary_writer.scalar(
                "steps_per_second", steps_per_second, step=global_step
            )
            start = time.time()
            
        if ((global_step + 1) % params["eval_steps"]) == 0:
            eval_callback(global_step)
            start = time.time()

        global_step += 1
        progbar.update(global_step)



if __name__ == "__main__":
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
    
    train_summary_writer = SummaryCallback(model_dir=params['model_dir'], mode="train")
    eval_summary_writer = SummaryCallback(model_dir=params['model_dir'], mode="eval")
    
    train_dataset_ = dataset_pipeline(train_features, train_labels, preprocess, params, mode="train")
    eval_dataset_ = dataset_pipeline(eval_features, eval_labels, preprocess, params, mode="eval")
    model_fn_ = LeNet(num_classes=43)
    loss_fn_ = loss()
    lr_schedular_fn = PolyDecaySchedular(
        learning_rate=params["poly_decay_schedular"]["learning_rate"],
        total_steps=params["train_steps"],
        learning_power=params["poly_decay_schedular"]["learning_power"],
        minimum_learning_rate=params["poly_decay_schedular"]["learning_rate_min"],
        end_learning_rate=params["poly_decay_schedular"]["end_learning_rate"],
        save_summary_steps=params["save_summary_steps"],
        train_summary_writer=train_summary_writer
    )
    eval_callback = eval(eval_dataset_, model_fn_, loss_fn_, eval_summary_writer, params)
    train_eval(train_dataset_, model_fn_, loss_fn_, lr_schedular_fn, eval_callback, train_summary_writer, params)
    