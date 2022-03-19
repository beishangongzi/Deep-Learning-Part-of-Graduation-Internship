import os

import numpy as np
import tensorflow as tf
from environs import Env

from sklearn.metrics import confusion_matrix, classification_report


def reports(X_test, y_test, model, target_names):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    classification = classification_report(y_test, y_pred, target_names=target_names)
    confusion = confusion_matrix(y_test, y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100

    return classification, confusion, Test_Loss, Test_accuracy


def predict_report(model_path, test_path):
    model_name = os.path.basename(model_path)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = np.array(test_ds.class_names)


    model = tf.keras.models.load_model(model_path)
    y_test = np.concatenate([y for x, y in test_ds], axis=0)
    X_test = np.concatenate([x for x, y in test_ds], axis=0)
    classification, confusion, Test_loss, Test_accuracy = reports(X_test, y_test, model, class_names)
    classification = str(classification)
    confusion = str(confusion)
    file_name = os.path.join(env.str("reports"), f'{model_name}.txt')
    with open(file_name, 'w') as x_file:
        x_file.write('{} Test loss (%)'.format(Test_loss))
        x_file.write('\n')
        x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))


if __name__ == '__main__':
    env = Env()
    env.read_env(".env")
    batch_size = env.int("batch_size", 32)
    img_height = env.int("img_height", 200)
    img_width = env.int("img_width", 200)
    test_dir = env.str("test_dir", "./../Data/Data_test")
    epochs = env.int("epochs")
    export_path = os.path.join(input("input the model name\n"))
    predict_report(export_path, test_dir)
