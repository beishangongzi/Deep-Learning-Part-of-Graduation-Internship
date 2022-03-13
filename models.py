import os

import keras
import pydotplus
import tensorflow as tf
import tensorflow_hub as hub
from environs import Env
from keras.applications.resnet import ResNet50, ResNet101
from keras.layers import Conv2D
from keras.utils.vis_utils import plot_model

env = Env()

from functools import wraps


def plot_file(logfile=env.str("plot_model", "image_of_model")):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            print(logfile)
            model = func(*args, **kwargs)
            keras.utils.vis_utils.pydot = pydotplus
            file_path = os.path.join(logfile, model.name + ".png")
            plot_model(model, file_path, show_shapes=True)
            return model

        return wrapped_function

    return logging_decorator


@plot_file()
def mobile_v3_transfer_model(pre_model):
    mobilenet_v2 = os.path.join(env.str("pre_model"), pre_model)

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


@plot_file()
def toy_res_net(pre_model):
    """
    即使没有使用pre_model,也要声明，保持接口一致
    """
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


@plot_file()
def toy_conv_net(pre_model):
    inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3), name="img")
    x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(int(os.getenv("num_class")))(x)
    model = keras.Model(inputs, outputs, name="toy_conv_net")
    model.summary()
    return model


@plot_file()
def toy_conv_net2(pre_model):
    inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3), name="img")
    x = keras.layers.Conv2D(10, 3, activation="relu")(inputs)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    out_puts = keras.layers.Dense(int(os.getenv("num_class")))(x)
    model = keras.Model(inputs, out_puts, name="toy_conv_net2")
    model.summary()
    return model


def restnet50(pre_model):
    base_model = ResNet50(weights="imagenet",
                          input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3), include_top=False)
    base_model.trainable = False
    inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAvgPool2D()(x)
    out_puts = keras.layers.Dense(int(os.getenv("num_class")))(x)
    model = keras.Model(inputs, out_puts, name="resnet50")
    model.summary()
    return model


def restnet101(pre_model):
    base_model = ResNet101(weights="imagenet",
                           input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
                           include_top=False)
    base_model.trainable = False
    inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAvgPool2D()(x)
    out_puts = keras.layers.Dense(int(os.getenv("num_class")))(x)
    model = keras.Model(inputs, out_puts, name="resnet50")
    return model
