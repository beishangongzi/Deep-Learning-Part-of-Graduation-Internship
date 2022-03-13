import datetime
import os
import time

import keras.callbacks
import tensorflow as tf
from keras import Model

import get_dataset


def training(model: Model, dataset=None):
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

    checkpoint_path = os.path.join("model_ckpt", model.name, "cp_{epoch:04d}.ckpt")
    save_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=False,
        save_freq=int(os.getenv("save_freq")) * int(os.getenv("batch_size"))
    )

    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=NUM_EPOCHS,
                        callbacks=[tensorboard_callback, save_callback])

    t = time.time()
    export_path = os.path.join("model", "{}-{}".format(model.name, int(t)))
    model.save(export_path)
    return export_path

