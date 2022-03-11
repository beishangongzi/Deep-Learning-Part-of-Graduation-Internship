import os
from abc import ABC

import tensorflow as tf
import tensorflow_hub as hub
from environs import Env
from keras import Model
from keras.layers import Conv2D, Flatten, Dense

env = Env()


# class _Model():
#     def __init__(self, class_names):
#         self.class_names_number = len(class_names)
#
#     def pre_model(self, pre_model) -> keras.Model:
#         # mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
#         mobilenet_v2 = "/media/andy/z/python/毕业实习/app/deep_learning/pre_model/" + pre_model
#
#         classifier_model = mobilenet_v2
#
#         feature_extractor_layer = hub.KerasLayer(
#             classifier_model,
#             input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
#             trainable=False)
#
#         model = tf.keras.Sequential([
#             # data_augmentation,
#             feature_extractor_layer,
#             tf.keras.layers.Dense(self.class_names_number)
#         ])
#
#         # model.summary()
#         return model
#
#     def res_net(self) -> keras.Model:
#         pass

class BaseModel(Model, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.d_last = Dense(7)

    @property
    def name(self):
        return self.__class__.__name__


class MyModel(BaseModel, ABC):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d_last(x)


class TransferLearningModel(BaseModel, ABC):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(7, activation="relu")

    def call(self, pre_model, training=None, mask=None):
        mobilenet_v2 = env.str("model_dir") + pre_model

        classifier_model = mobilenet_v2

        feature_extractor_layer = hub.KerasLayer(
            classifier_model,
            input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
            trainable=False)

        model = tf.keras.Sequential([
            # data_augmentation,
            feature_extractor_layer,
            self.d_last
        ])

        return model
