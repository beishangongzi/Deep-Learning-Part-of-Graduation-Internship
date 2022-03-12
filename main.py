import os

from dotenv import load_dotenv
import tensorflow as tf

from models import *
from training import training
from predict import predict_report
from visual_show import predict_visuality

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('pre_model', "tf2-preview_mobilenet_v2_classification_4", 'the pre model')
tf.compat.v1.flags.DEFINE_string("model", "mobile_v3_transfer_model", "model that used")

def main(unused_argv):
    load_dotenv()
    model = eval(FLAGS.model)(FLAGS.pre_model)
    dataset = os.getenv("data_root")
    export_path = training(model, dataset)
    predict_report(export_path, os.getenv("test_dir"))
    predict_visuality(export_path, os.getenv("test_dir"))


if __name__ == '__main__':
    tf.compat.v1.app.run()
