import keras
from keras import losses
from src.behavioural_cloning.params import params
from src.behavioural_cloning.model import Model, load_weights
from src.behavioural_cloning.generator import keras_generator
from src.behavioural_cloning.callbacks import TrainingCallback, SnapshotCallback, TensorBoardCallback, \
    ValidationCallback, LearningRateSchedular, poly_decay

driving_log_path = ["./data/driving_log.csv", "./data/driving_log2.csv"]


params, train_generator, eval_generator, test_generator = keras_generator(
        params,
        driving_log_path_list=driving_log_path,
)

model_builder = Model(weight_decay=params["weight_decay"])
model = model_builder()
model = load_weights(model, model_weight_path=params["model_weight_path"])
print("Model: \n\t Input: {}, Output: {}".format(model.inputs, model.outputs))

poly_decay_fn = poly_decay(
        initial_lrate=params["poly_decay"]["learning_rate"],
        total_steps=params["epochs"]*len(train_generator),
        end_learning_rate=params["poly_decay"]["end_learning_rate"],
        learning_power=params["poly_decay"]["learning_power"],
        learning_rate_min=params["poly_decay"]["learning_rate_min"]
)


tensorboard_callback = TensorBoardCallback(params)()
snapshot_callback = SnapshotCallback(params)
training_callback = TrainingCallback(tensorboard_callback, params)
learning_rate_callback = LearningRateSchedular(poly_decay_fn, tensorboard_callback, params)
validation_callback = ValidationCallback(tensorboard_callback, model, eval_generator, params)


model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss=losses.mean_squared_error
)

model.fit_generator(
    train_generator,
    epochs=params["epochs"],
    steps_per_epoch=len(train_generator),
    callbacks=[tensorboard_callback, snapshot_callback, training_callback, validation_callback, learning_rate_callback]
)

