import numpy as np
import tensorflow as tf


class Model:
    def __init__(self):
        self.base_model = tf.keras.applications.MobileNetV2(
                input_shape=(120, 320, 3),
                include_top=False,
                weights='imagenet'
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)
        
    def __call__(self):
        inputs = self.base_model.inputs
        outputs = self.base_model.outputs
        outlayer = self.flatten(outputs[0])
        outlayer = self.dense(outlayer)
        model = tf.keras.Model(
                inputs=inputs,
                outputs=outlayer,
                name="mobile_net_augmented"
        )
        return model
    
    
if __name__ == "__main__":
    model = Model()
    out = model()
    print(out.inputs, out.outputs)
    