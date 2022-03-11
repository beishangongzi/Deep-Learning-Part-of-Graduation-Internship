import os

from dotenv import load_dotenv

from models import *
from training import training
from predict import predict_report
from visual_show import predict_visuality


def main():
    load_dotenv()
    transfer_learning_model = TransferLearningModel()
    model_name =  transfer_learning_model.name
    model = transfer_learning_model("tf2-preview_mobilenet_v2_classification_4")

    dataset = os.getenv("data_root")
    export_path = training(model, dataset, name=model_name)
    predict_report(export_path, os.getenv("test_dir"))
    predict_visuality(export_path, os.getenv("test_dir"))


if __name__ == '__main__':
    main()