import os
import pathlib

from dotenv import load_dotenv

from models import *
from predict import predict_report
from training import training
from visual_show import predict_visuality

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('pre_model', "tf2-preview_mobilenet_v2_classification_4", 'the pre model')
tf.compat.v1.flags.DEFINE_string("model", "mobile_v3_transfer_model", "model that used")
tf.compat.v1.flags.DEFINE_bool("train", True, "train a new model or use saved model")


def begin_main():
    if not os.path.exists("reports"):
        os.mkdir("reports")
    if not os.path.exists("visual_show_images"):
        os.mkdir("visual_show_images")
    if not os.path.exists("model"):
        os.mkdir("model")
    if not os.path.exists("pre_model"):
        os.mkdir("pre_model")
    if not os.path.exists("logs/fit"):
        os.makedirs("logs/fit")
    if not os.path.exists("model_ckpt"):
        os.makedirs("model_ckpt")


def main(unused_argv):
    load_dotenv()
    begin_main()
    if FLAGS.train:
        model = eval(FLAGS.model)(FLAGS.pre_model)

        dataset = pathlib.Path(os.getenv("data_root"))
        export_path = training(model, dataset)
    else:
        export_path = os.path.join("model", FLAGS.model)
    predict_report(export_path, pathlib.Path(os.getenv("test_dir")))
    predict_visuality(export_path, pathlib.Path(os.getenv("test_dir")))


if __name__ == '__main__':
    tf.compat.v1.app.run()
