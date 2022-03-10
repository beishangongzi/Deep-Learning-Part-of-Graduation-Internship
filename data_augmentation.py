from tensorflow import keras
import tensorflow as tf


def random_invert_img(x, p=0.5):
    if tf.random.uniform([]) < p:
        x = (255 - x)
    else:
        x
    return x


def random_invert(factor=0.5):
    return keras.layers.Lambda(lambda x: random_invert_img(x, factor))


class RandomInvert(keras.layers.Layer):
    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return random_invert_img(x)


data_augmentation = tf.keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.2),
    # RandomInvert()
])
