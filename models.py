import os

import keras
import tensorflow as tf
import tensorflow_hub as hub
from environs import Env
from keras.layers import Conv2D

env = Env()


def mobile_v3_transfer_model(pre_model):
    mobilenet_v2 = env.str("model_dir") + pre_model

    classifier_model = mobilenet_v2

    feature_extractor_layer = hub.KerasLayer(
        classifier_model,
        input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
        trainable=False)

    model = tf.keras.Sequential([
        # data_augmentation,
        feature_extractor_layer,
        keras.layers.Dense(int(os.getenv("num_class")))
    ], name="mobile_v3_transfer_model-" + pre_model)

    return model


def toy_res_net():
    inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3), name="img")
    x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    block_1_output = keras.layers.MaxPooling2D(3)(x)

    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = keras.layers.add([x, block_1_output])

    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = keras.layers.add([x, block_2_output])

    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_3_output)
    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(int(os.getenv("num_class")))(x)
    model = keras.Model(inputs, outputs, name="toy_resnet")
    model.summary()
    return model
