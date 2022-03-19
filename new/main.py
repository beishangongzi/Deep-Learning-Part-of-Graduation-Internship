import datetime
import os

import tensorflow as tf
from environs import Env
from keras import layers
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from tensorflow import keras


def main():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential(name="vgg16_new")
    model.add(data_augmentation)
    model.add(Rescaling(scale=1.0 / 255))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.summary()
    checkpoint_path = os.path.join(env.str("model_dir"), model.name, "_cp_best.ckpt")
    save_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=False,
        # save_freq=int(os.getenv("save_freq")) * int(os.getenv("batch_size"))
        save_best_only=True
    )
    log_dir = os.path.join(env.str("log_dir"), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1)  # Enable histogram computation for every epoch.

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[save_callback, tensorboard_callback]
    )
    export_path = os.path.join(env.str("model_dir"), model.name, "{best}")
    model.save(export_path)
    print(export_path)


if __name__ == '__main__':
    env = Env()
    env.read_env(".env")
    batch_size = env.int("batch_size", 32)
    img_height = env.int("img_height", 200)
    img_width = env.int("img_width", 200)
    data_dir = env.str("dir_dir", "./../Data/Data")
    epochs = env.int("epochs")
    main()
