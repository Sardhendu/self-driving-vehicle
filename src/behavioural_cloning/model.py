import keras
import tensorflow as tf
from keras import losses
from keras.applications.xception import Xception

from keras.layers import Cropping2D


class Model:
    def __init__(self, weight_decay=None):
        """
        # The range of labels (steeting) in good driving condition goes from -1 to 1. Now this could be a good use
        case for tanh.
        """
        self.base_model = Xception(weights='imagenet',
                                   input_shape=(90, 320, 3),
                                   include_top=False)  # imports the mobilenet model and discards the last 1000
        # neuron layer.
        if weight_decay is not None:
            self.add_weight_decay(weight_decay)
        self.crop = Cropping2D(
                cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3))
        self.normalize = keras.layers.Lambda(lambda x: (x/255.)-0.5)
        self.flatten = keras.layers.Flatten()
        self.dense_squash = keras.layers.Dense(1, activation="tanh")
    
    def __call__(self):
        inputs = keras.layers.Input(shape=(160, 320, 3))
        inputs_pp = self.crop(inputs)
        inputs_pp = self.normalize(inputs_pp)
        inputs_pp = self.base_model(inputs_pp)
        inputs_pp = self.flatten(inputs_pp)
        outlayer = self.dense_squash(inputs_pp)
        
        model = keras.Model(
                inputs=[inputs],
                outputs=[outlayer],
                name="xception_augmented"
        )
        print(self.base_model.summary())
        return model
    
    def add_weight_decay(self, decay_val):
        for layer in self.base_model.layers:
            print("layer: ", layer)
            if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                layer.add_loss(lambda: keras.regularizers.l2(decay_val)(layer.kernel))
            if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                layer.add_loss(lambda: keras.regularizers.l2(decay_val)(layer.bias))


def load_weights(model, model_weight_path):
    if model_weight_path is not None:
        print('[Weights] Loading model weights from file : ', model_weight_path)
        model.load_weights(model_weight_path, by_name=True)
    return model


if __name__ == "__main__":
    model_builder = Model()
    model = model_builder()
    print("Model: \n\t Input: {}, Output: {}".format(model.inputs, model.outputs))
    
    model.compile(
            optimizer=keras.optimizers.Adam(0.01),
            loss=losses.mean_squared_error
    )

# import numpy as np
# import tensorflow as tf
# a = tf.constant(np.random.random((1, 160, 320, 3)), dtype=tf.float32)
#
# b = tf.image.crop_to_bounding_box(
#     a,
#     offset_height=40,
#     offset_width=0,
#     target_height=120,
#     target_width=320
# )
#
#
# print(b.shape)