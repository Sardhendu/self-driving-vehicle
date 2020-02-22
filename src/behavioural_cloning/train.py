import tensorflow as tf

from src.behavioural_cloning.params import params
from src.behavioural_cloning.model import Model
from src.behavioural_cloning.generator import tf_generator
from src.behavioural_cloning.callbacks import TrainingCallback, SnapshotCallback, TensorBoardCallback, ValidationCallback


driving_log_path = "./data/driving_log.csv"


params, train_generator, eval_generator = tf_generator(
        params,
        driving_log_path=driving_log_path,
        mode="train"
)
from pprint import pprint
pprint(params)
model = Model()()

tensorboard_callback = TensorBoardCallback(params)()
snapshot_callback = SnapshotCallback(params)
training_callback = TrainingCallback(params)
validation_callback = ValidationCallback(model, eval_generator, params)

model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.MeanSquaredError()
)
model.fit_generator(
        train_generator,
        epochs=3,
        validation_data=eval_generator,
        callbacks=[tensorboard_callback, snapshot_callback, training_callback, validation_callback]
)
