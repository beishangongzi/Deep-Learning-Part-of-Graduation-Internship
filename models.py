from enum import Enum

import datetime
import os
import time

import keras
import tensorflow as tf
import tensorflow_hub as hub

import get_dataset
from data_augmentation import data_augmentation


class Model():
    def __init__(self, class_names):
        self.class_names_number = len(class_names)

    def pre_model(self, pre_model) -> keras.Model:
        # mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
        mobilenet_v2 = "/media/andy/z/python/毕业实习/app/deep_learning/pre_model/" + pre_model

        classifier_model = mobilenet_v2

        feature_extractor_layer = hub.KerasLayer(
            classifier_model,
            input_shape=(int(os.getenv("img_height")), int(os.getenv("img_width")), 3),
            trainable=False)

        model = tf.keras.Sequential([
            # data_augmentation,
            feature_extractor_layer,
            tf.keras.layers.Dense(self.class_names_number)
        ])

        # model.summary()
        return model

    def res_net(self) -> keras.Model:
        pass
