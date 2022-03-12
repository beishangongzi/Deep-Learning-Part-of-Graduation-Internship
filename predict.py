import os

import numpy as np
import tensorflow as tf

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
        test_path, labels='inferred', label_mode='int',
        class_names=None, color_mode='rgb', batch_size=32, image_size=(int(os.getenv("img_height")),
                                                                       int(os.getenv("img_width"))), shuffle=False,
        seed=None,
        validation_split=None, subset=None,
        interpolation='bilinear', follow_links=False,
        crop_to_aspect_ratio=False,
    )
    class_names = np.array(test_ds.class_names)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))  # Where x—images, y—labels.

    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.models.load_model(model_path)
    y_test = np.concatenate([y for x, y in test_ds], axis=0)
    X_test = np.concatenate([x for x, y in test_ds], axis=0)
    classification, confusion, Test_loss, Test_accuracy = reports(X_test, y_test, model, class_names)
    classification = str(classification)
    confusion = str(confusion)
    file_name = os.path.join(os.getenv("reports"), f'{model_name}.txt')
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
    from dotenv import load_dotenv

    load_dotenv()
    export_path = os.path.join(os.getenv("model"), input("input the model name\n"))
    predict_report(export_path, os.getenv("data_root"))
