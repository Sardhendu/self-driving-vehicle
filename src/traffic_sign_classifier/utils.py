import tensorflow as tf
from typing import Any


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


class PolyDecaySchedular(tf.optimizers.schedules.LearningRateSchedule):
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
            (self.learning_rate - self.end_learning_rate) * tf.pow(1 - ((current_step) / self.total_steps), self.learning_power) + self.end_learning_rate,
            dtype=tf.float32
        )
        nw_lr = tf.maximum(lrate, self.minimum_learning_rate)
        
        if ((step+1) % self.save_summary_steps) == 0:
            self.train_summary_writer.scalar("learning_rate", nw_lr, step)
        return nw_lr
