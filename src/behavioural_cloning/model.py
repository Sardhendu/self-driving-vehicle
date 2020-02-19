import numpy as np
import tensorflow as tf


class Model:
    def __init__(self):
        """
        # The range of labels (steeting) in good driving condition goes from -1 to 1. Now this could be a good use
        case for tanh.
        """
        self.base_model = tf.keras.applications.MobileNetV2(
                input_shape=(120, 320, 3),
                include_top=False,
                weights='imagenet'
        )

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)
        self.squash = tf.nn.tanh
        
    def __call__(self):
        inputs = self.base_model.inputs
        outputs = self.base_model.outputs
        outlayer = self.flatten(outputs[0])
        outlayer = self.dense(outlayer)
        outlayer = self.squash(outlayer)
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

    model.compile(
            optimizer=tf.keras.optimizers.Adam(0.01),
            loss=tf.keras.losses.MSE()
    )
    model.fit()
    
    
    