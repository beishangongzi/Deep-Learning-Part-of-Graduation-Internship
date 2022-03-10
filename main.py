import datetime
import os
import time

import tensorflow as tf

import get_dataset
from models import Model


def main():
    train_ds, val_ds, class_names = get_dataset.get_dataset(str(os.getenv("data_root")))
    pre_model = input("input model name")
    model = Model(class_names).pre_model(pre_model=pre_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)  # Enable histogram computation for every epoch.

    NUM_EPOCHS = 20

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=NUM_EPOCHS,
                        callbacks=tensorboard_callback)

    t = time.time()
    export_path = "model/{}-{}".format(pre_model, int(t))
    model.save(export_path)


if __name__ == '__main__':
    main()
