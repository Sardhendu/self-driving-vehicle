
import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(input_shape=(120, 320, 3),
                                               include_top=False,
                                               weights='imagenet')

print(base_model.inputs, base_model.outputs)
