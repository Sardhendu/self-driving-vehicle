
import tensorflow as tf
import time
from typing import Callable, Dict, Any
import os
from src import commons
from src.traffic_sign_classifier import ops
from src.traffic_sign_classifier.params import params
from src.traffic_sign_classifier.preprocess import preprocess
from src.traffic_sign_classifier.model import dataset_pipeline, LeNet
strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")


def loss():
    def loss_(y_true: tf.Tensor, y_logits: tf.Tensor):
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
        eval_summary_writer: ops.SummaryCallback,
        params: Dict
):
    eval_loss = tf.keras.metrics.Sum("eval_loss", dtype=tf.float32)
    eval_acc = tf.keras.metrics.Accuracy("eval_accuracy", dtype=tf.float32)
    eval_pr = ops.PrecisionRecall(params["num_classes"], threshold=None, summary_dir=params["summary_dir"])
    
    def eval_(global_step):
        iterator_ = iter(eval_dataset)
        progbar = tf.keras.utils.Progbar(params["eval_data_cnt"])
        eval_loss.reset_states()
        eval_acc.reset_states()
        eval_pr.reset_states()
        
        it = 0
        while it < params["eval_data_cnt"]:
            eval_images, eval_target_one_hot = next(iterator_)
            eval_pred_logits = model_builder(eval_images)
            eval_pred_probs = tf.nn.softmax(eval_pred_logits)

            loss_eval = loss_fn(eval_target_one_hot, eval_pred_logits)
            eval_loss.update_state(loss_eval)

            labels = tf.argmax(eval_target_one_hot, axis=-1)
            preds = tf.argmax(eval_pred_logits, axis=-1)
            eval_acc.update_state(y_true=labels, y_pred=preds)
            eval_pr.update_state(labels, eval_pred_probs)

            it += 1
            progbar.update(it)
        avg_eval_loss = tf.divide(eval_loss.result(), params["eval_data_cnt"])
        eval_summary_writer.scalar("loss", avg_eval_loss, global_step)
        eval_summary_writer.scalar("eval_accuracy", eval_acc.result(), global_step)
        eval_pr.write_summary(global_step)

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
    optimizer_ = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedular)
    checkpoints = ops.CheckpointCallback(model_dir=params["model_dir"], optimizer=optimizer_, model=model_builder)

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

        if ((global_step + 1) % params["save_summary_steps"]) == 0 or global_step == 1:
            end = time.time()
            avg_train_loss = tf.divide(train_loss.result(), params["save_summary_steps"])
            train_summary_writer.scalar("loss", avg_train_loss, global_step)
            train_loss.reset_states()
            steps_per_second = params["save_summary_steps"] / (end - start)
            train_summary_writer.scalar(
                "steps_per_second", steps_per_second, step=global_step
            )
            checkpoints.save(global_step)
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
    
    train_summary_writer = ops.SummaryCallback(summary_dir=os.path.join(params['summary_dir'], "train"))
    eval_summary_writer = ops.SummaryCallback(summary_dir=os.path.join(params['summary_dir'], "eval"))

    train_preprocess = preprocess(mode="train")
    eval_preprocess = preprocess(mode="eval")
    train_dataset_ = dataset_pipeline(train_features, train_labels, train_preprocess, params, mode="train")
    eval_dataset_ = dataset_pipeline(eval_features, eval_labels, eval_preprocess, params, mode="eval")
    model_fn_ = LeNet(num_classes=43)
    loss_fn_ = loss()

    lr_schedular_fn = ops.poly_cosine_schedular(params, train_summary_writer)
    eval_callback = eval(eval_dataset_, model_fn_, loss_fn_, eval_summary_writer, params)
    train_eval(train_dataset_, model_fn_, loss_fn_, lr_schedular_fn, eval_callback, train_summary_writer, params)


# https://docs.w3cub.com/tensorflow~python/tf/keras/preprocessing/image/random_shift/