import tensorflow as tf


class PolyDecaySchedular(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr, total_steps, learning_power, min_lr, end_learning_rate):
        self.lr = lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.learning_power = learning_power
        self.end_learning_rate = end_learning_rate

    # def __call__(self, step):
    #     return tf.cond(step < self.warmup_steps, lambda: self.warmup_lr, lambda: self.lr)

    def __call__(self, step):
        global_step = tf.minimum(step, self.total_steps)
        lrate = tf.cast(
            (self.lr - self.end_learning_rate) * tf.pow(1 - ((global_step) / self.total_steps), self.learning_power) + self.end_learning_rate,
            dtype=tf.float32
        )
        return tf.maximum(lrate, self.min_lr)
