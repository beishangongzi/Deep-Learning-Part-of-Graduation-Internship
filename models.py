import os
from abc import ABC
from functools import wraps

import keras
import pydotplus
import tensorflow as tf
import tensorflow_hub as hub
from keras.applications.resnet import ResNet50, ResNet101
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense, Add
from keras.models import Model
from keras.utils.vis_utils import plot_model


def plot_file(logfile=os.getenv("plot_model_dir")):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            print(logfile)
            model = func(*args, **kwargs)
            keras.utils.vis_utils.pydot = pydotplus
            file_path = os.path.join(logfile, model.name + ".png")
            try:
                plot_model(model, file_path, show_shapes=True)
            except ImportError as e:
                print(e)
            return model

        return wrapped_function

    return logging_decorator


@plot_file()
def mobile_v3_transfer_model(pre_model):
    mobilenet_v2 = os.path.join(os.getenv("pre_model_dir"), pre_model)

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


@plot_file()
def restnet50(pre_model):
    base_model = ResNet50(weights="imagenet",
                          input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3), include_top=False)
    flag = False if os.getenv("transfer_learning_trainable") == "False" else True
    base_model.trainable = flag
    inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3))
    x = base_model(inputs, training=flag)
    x = keras.layers.GlobalAvgPool2D()(x)
    out_puts = keras.layers.Dense(int(os.getenv("num_class")))(x)
    model = keras.Model(inputs, out_puts, name="resnet50")
    model.summary()
    return model


@plot_file()
def restnet101(pre_model):
    base_model = ResNet101(weights="imagenet",
                           input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
                           include_top=False)
    base_model.trainable = False if os.getenv("transfer_learning_trainable") == "False" else True
    inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3))
    x = base_model(inputs, training=False if os.getenv("transfer_learning_trainable") == "False" else True)
    x = keras.layers.GlobalAvgPool2D()(x)
    out_puts = keras.layers.Dense(int(os.getenv("num_class")))(x)
    model = keras.Model(inputs, out_puts, name="resnet101")
    return model


@plot_file()
def vgg16(pre_model):
    base_model = VGG16(weights="imagenet",
                       input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
                       include_top=False)
    base_model.trainable = False if os.getenv("transfer_learning_trainable") == "False" else True
    inputs = keras.Input(shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3))
    x = base_model(inputs, training=False if os.getenv("transfer_learning_trainable") == "False" else True)
    x = keras.layers.GlobalAvgPool2D()(x)
    out_puts = keras.layers.Dense(int(os.getenv("num_class")), activation="softmax")(x)
    model = keras.Model(inputs, out_puts, name="vgg16")
    return model


@plot_file()
def restnet18(pre_model):
    class ResnetBlock(Model):
        """
        A standard resnet block.
        """

        def __init__(self, channels: int, down_sample=False):
            """
            channels: same as number of convolution kernels
            """
            super().__init__()

            self.__channels = channels
            self.__down_sample = down_sample
            self.__strides = [2, 1] if down_sample else [1, 1]

            KERNEL_SIZE = (3, 3)
            # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
            INIT_SCHEME = "he_normal"

            self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                                 kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
            self.bn_1 = BatchNormalization()
            self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                                 kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
            self.bn_2 = BatchNormalization()
            self.merge = Add()

            if self.__down_sample:
                # perform down sampling using stride of 2, according to [1].
                self.res_conv = Conv2D(
                    self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
                self.res_bn = BatchNormalization()

        def call(self, inputs, training=None, mask=None):
            res = inputs

            x = self.conv_1(inputs)
            x = self.bn_1(x)
            x = tf.nn.relu(x)
            x = self.conv_2(x)
            x = self.bn_2(x)

            if self.__down_sample:
                res = self.res_conv(res)
                res = self.res_bn(res)

            # if not perform down sample, then add a shortcut directly
            x = self.merge([x, res])
            out = tf.nn.relu(x)
            return out

    class ResNet18(Model):

        def __init__(self, num_classes, **kwargs):
            """
                num_classes: number of classes in specific classification task.
            """
            super().__init__(**kwargs)
            self.conv_1 = Conv2D(64, (7, 7), strides=2,
                                 padding="same", kernel_initializer="he_normal")
            self.init_bn = BatchNormalization()
            self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
            self.res_1_1 = ResnetBlock(64)
            self.res_1_2 = ResnetBlock(64)
            self.res_2_1 = ResnetBlock(128, down_sample=True)
            self.res_2_2 = ResnetBlock(128)
            self.res_3_1 = ResnetBlock(256, down_sample=True)
            self.res_3_2 = ResnetBlock(256)
            self.res_4_1 = ResnetBlock(512, down_sample=True)
            self.res_4_2 = ResnetBlock(512)
            self.avg_pool = GlobalAveragePooling2D()
            self.flat = Flatten()
            self.fc = Dense(num_classes, activation="softmax")

        def call(self, inputs, training=None, mask=None):
            out = self.conv_1(inputs)
            out = self.init_bn(out)
            out = tf.nn.relu(out)
            out = self.pool_2(out)
            for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2,
                              self.res_4_1, self.res_4_2]:
                out = res_block(out)
            out = self.avg_pool(out)
            out = self.flat(out)
            out = self.fc(out)
            return out

    model = ResNet18(int(os.getenv("num_class")))
    model.build(input_shape=(None, int(os.getenv("img_height")), int(os.getenv("img_width")), 3))
    model.summary()
    print(model.name)
    return model
