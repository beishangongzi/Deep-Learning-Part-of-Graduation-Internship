import os

import keras
import tensorflow as tf
from dotenv import load_dotenv


def get_model(path_or_name):
    if type(path_or_name) is str:
        model = tf.keras.models.load_model(path_or_name)
        name = os.path.basename(path_or_name)
    if type(path_or_name) is keras.Model:
        model = path_or_name
    if type(path_or_name) is keras.Sequential:
        return path_or_name
    raise Exception("can not find any model")
    
    
if __name__ == '__main__':
    load_dotenv()
    p1 = "/media/andy/z/python/毕业实习/app/deep_learning/model/1646726472"
    print(get_model(p1).name)
    from deep_learning.models import TransferLearningModel
    m = TransferLearningModel()("tf2-preview_mobilenet_v2_classification_4")
    print(type(m))
    print(get_model(m).name)