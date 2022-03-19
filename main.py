import os

from dotenv import load_dotenv

load_dotenv()

import pathlib

from models import *
from predict import predict_report
from training import training
from visual_show import predict_visuality


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
    if not os.path.exists("daemon"):
        os.mkdir("daemon")


def main(unused_argv):
    begin_main()
    if FLAGS.train:
        print(FLAGS.model)
        model = eval(FLAGS.model)(FLAGS.pre_model)
        if FLAGS.continue_train is not None:
            pre_weights = FLAGS.continue_train
            if os.path.exists(os.path.join(os.getenv("model_dir"), pre_weights)):
                pre_weights = os.path.join(os.getenv("model_dir", pre_weights))
            else:
                pre_weights = os.path.join(os.getenv("save_in_training"), model.name, pre_weights)
                print(f"load weights from {pre_weights}")
            model.load_weights(pre_weights)
        dataset = pathlib.Path(os.getenv("data_root"))
        export_path = training(model, dataset)
    else:
        export_path = os.path.join(os.getenv("model_dir"), FLAGS.model)
    predict_report(export_path, pathlib.Path(os.getenv("test_dir")))
    predict_visuality(export_path, pathlib.Path(os.getenv("test_dir")))


if __name__ == '__main__':
    FLAGS = tf.compat.v1.flags.FLAGS
    tf.compat.v1.flags.DEFINE_string('pre_model', os.getenv("pre_model"), 'the pre model)
    tf.compat.v1.flags.DEFINE_string("model", os.getenv("model"), "model that used")
    tf.compat.v1.flags.DEFINE_bool("train", False if os.getenv("train") == "False" else True,
                                   "train a new model or use saved model")
    tf.compat.v1.flags.DEFINE_string("continue_train", os.getenv("continue_train"), "continue train")
    tf.compat.v1.app.run()
