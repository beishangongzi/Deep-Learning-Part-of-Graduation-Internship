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
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss="categorical_crossentropy",
        metrics=['acc', "mae"])

    log_dir = os.path.join(os.getenv("log_dir"), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)  # Enable histogram computation for every epoch.
    t = datetime.datetime.now().day.__str__() + "-" + datetime.datetime.now().hour.__str__() + datetime.datetime.now().minute.__str__()
    checkpoint_path = os.path.join("model_ckpt", model.name, t + "_cp_{epoch:04d}.ckpt")
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

