import numpy as np
import tensorflow as tf
from typing import Any


class PolyScaledCosineAnneling(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            learning_rate: int,
            total_steps: int,
            poly_power: int,
            end_learning_rate: int,
            save_summary_steps: int,
            train_summary_writer: Any
    ):
        self.learning_rate = learning_rate
        self.poly_power = poly_power
        self.total_steps = total_steps
        self.end_learning_rate = end_learning_rate
        self.save_summary_steps = save_summary_steps
        self.train_summary_writer = train_summary_writer
    
    def __call__(self, step):
        # This is polynomial decay
        # We do this to dynamically scale the cosine function and to adjust the learning_rate decay between (
        # learning_rate,
        # minimum_learning_rate)
        decay_factor = tf.cast(1 - (0.5 * tf.pow(1 - (step / self.total_steps), self.poly_power)), dtype=tf.float32)
        
        nw_lr = (
                self.learning_rate * decay_factor
                * ((1 + self.end_learning_rate) + tf.cos(np.pi * tf.cast(step, dtype=tf.float32) / self.total_steps))
        )
        
        if ((step +1) % self.save_summary_steps) == 0:
            self.train_summary_writer.scalar("learning_rate", nw_lr, step)
        return nw_lr


class PolyDecayScheduler(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            learning_rate: int,
            total_steps: int,
            learning_power: int,
            minimum_learning_rate: int,
            end_learning_rate: int,
            save_summary_steps: int,
            train_summary_writer: Any
    ):
        self.learning_rate = learning_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.total_steps = total_steps
        self.learning_power = learning_power
        self.end_learning_rate = end_learning_rate
        self.train_summary_writer = train_summary_writer
        self.save_summary_steps = save_summary_steps
    
    def __call__(self, step):
        current_step = tf.minimum(step, self.total_steps)
        lrate = tf.cast(
                (self.learning_rate - self.end_learning_rate) * tf.pow(1 - ((current_step) / self.total_steps),
                                                                       self.learning_power) + self.end_learning_rate,
                dtype=tf.float32
        )
        nw_lr = tf.maximum(lrate, self.minimum_learning_rate)
        
        if ((step + 1) % self.save_summary_steps) == 0:
            self.train_summary_writer.scalar("learning_rate", nw_lr, step)
        return nw_lr


def poly_decay_schedular(params, train_summary_writer):
    return PolyDecayScheduler(
            learning_rate=params["poly_decay_schedular"]["learning_rate"],
            total_steps=params["train_steps"]*params["epochs"],
            learning_power=params["poly_decay_schedular"]["learning_power"],
            minimum_learning_rate=params["poly_decay_schedular"]["learning_rate_min"],
            end_learning_rate=params["poly_decay_schedular"]["end_learning_rate"],
            save_summary_steps=params["save_summary_steps"],
            train_summary_writer=train_summary_writer
        )


def poly_cosine_schedular(params, train_summary_writer):
    return PolyScaledCosineAnneling(
            learning_rate=params["poly_cosine_schedular"]["learning_rate"],
            total_steps=params["train_steps"]*params["epochs"],
            poly_power=params["poly_cosine_schedular"]["poly_power"],
            end_learning_rate=params["poly_cosine_schedular"]["end_learning_rate"],
            save_summary_steps=params["save_summary_steps"],
            train_summary_writer=train_summary_writer
    )


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


class CheckpointCallback:
    def __init__(self, model_dir, optimizer, model):
        self.model_dir = model_dir
        self.ckpt = tf.train.Checkpoint(optimizer=optimizer, net=model)
        self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, model_dir, max_to_keep=3, keep_checkpoint_every_n_hours=3
        )
        
    def save(self, step):
        self.ckpt_manager.save(step)
        print("Saved checkpoint for step {}: {}".format(int(step), self.model_dir))
        
    def restore(self):
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
