import datetime
import os
import time

import tensorflow as tf
from keras import Model

import get_dataset


def training(model: Model, dataset=None, name=None):
    if dataset is None:
        dataset = os.getenv("data_root")
    if dataset is None:
        raise Exception("Dataset is not set")
    train_ds, val_ds, class_names = get_dataset.get_dataset(dataset)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    log_dir = os.path.join(os.getenv("log_dir"), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)  # Enable histogram computation for every epoch.

    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=NUM_EPOCHS,
                        callbacks=tensorboard_callback)

    t = time.time()
    export_path = "model/{}-{}".format(model.name if name is None else name, int(t))
    model.save(export_path)
    return export_path

