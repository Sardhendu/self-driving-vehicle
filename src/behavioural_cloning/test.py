from sklearn import metrics
from src.behavioural_cloning.params import params
from src.behavioural_cloning.model import Model, load_weights
from src.behavioural_cloning.generator import keras_generator

driving_log_path = ["./data/driving_log.csv", "./data/driving_log2.csv"]

params, train_generator, eval_generator, test_generator = keras_generator(
        params,
        driving_log_path_list=driving_log_path,
)

model_builder = Model(weight_decay=params["weight_decay"])
model = model_builder()
model = load_weights(model, model_weight_path=params["model_weight_path"])
print("Model: \n\t Input: {}, Output: {}".format(model.inputs, model.outputs))

final_mse = 0
for num_ in range(0, len(test_generator)):
    data_, steering_val = test_generator[num_]
    out_steer_val = model.predict_on_batch(data_)
    mse_loss = metrics.mean_squared_error(steering_val.flatten(), out_steer_val.flatten())
    final_mse += mse_loss
    
print("Test MSE: --> ", final_mse/len(test_generator))





