import os

import numpy as np
import tensorflow as tf


def get_dataset(data_root):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_root),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(int(os.getenv("img_height")), int(os.getenv("img_width"))),
        batch_size=int(os.getenv("batch_size"))
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_root),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(int(os.getenv("img_height")), int(os.getenv("img_width"))),
        batch_size=int(os.getenv("batch_size"))
    )
    class_names = np.array(train_ds.class_names)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))  # Where x—images, y—labels.
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names
