import os

from keras import layers
from tensorflow import keras

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(int(os.getenv("img_height")),
                                       int(os.getenv("img_width")),
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)
