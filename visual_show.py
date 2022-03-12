import os

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from matplotlib import pyplot as plt




def predict_visuality(model_path, test_path):
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

    for image_batch, labels_batch in test_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    result_batch = model.predict(test_ds)
    predicted_class_names = class_names[tf.math.argmax(result_batch, axis=-1)]
    true_class_name = class_names[labels_batch]

    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)

    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        if true_class_name[n] == predicted_class_names[n]:
            plt.title(true_class_name[n])
        else:
            plt.title(true_class_name[n] + " | " + predicted_class_names[n])
        plt.axis('off')
    _ = plt.suptitle("ImageNet predictions", )
    plt.savefig(os.path.join("visual_show_images",f"{os.path.basename(model_name)}.png"))


if __name__ == '__main__':
    export_path = os.path.join(os.getenv("model"), input("input the model name\n"))
    load_dotenv()
    predict_visuality(export_path, os.getenv("data_root"))
