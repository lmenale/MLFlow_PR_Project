import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

import os


def load_data(data_dir, image_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size,
    )
    return train_ds, val_ds

def preprocess(train_ds, val_ds, test_ratio):
    val_batches = val_ds.cardinality()
    test_ds = val_ds.take(val_batches // test_ratio)
    val_ds = val_ds.skip(val_batches // test_ratio)
    print("Number of validation batches: %d" % val_ds.cardinality())
    print("Number of test batches: %d" % test_ds.cardinality())
    
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def build_model(input_shape, num_classes, base_learning_rate) -> tf.keras.Model:
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    global_average_layer = layers.GlobalAveragePooling2D()
    prediction_layer = layers.Dense(num_classes, activation="softmax")
    
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    
    return model

def train_model(model, train_ds, val_ds, initial_epochs, callbacks):
    history = model.fit(
        train_ds,
        epochs=initial_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )
    return history

def fine_tune_model(model, base_model, fine_tune_at, base_learning_rate, fine_tune_epochs, callbacks):
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    
    history_fine = model.fit(
        train_ds,
        epochs=fine_tune_epochs,
        initial_epoch=len(history.epoch),
        validation_data=val_ds,
        callbacks=callbacks,
    )
    
    return history_fine

def save_model(model, filename):
    model.save(filename)


if __name__ == "__main__":
    # Constants
    BATCH_SIZE = 32
    IMAGE_SIZE = (180, 180)
    DATA_DIR = "data/raw"
    TEST_RATIO = 5
    BASE_LEARNING_RATE = 0.0001
    FINE_TUNE_AT = 100
    INITIAL_EPOCHS = 10
    FINE_TUNE_EPOCHS = 10
    
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Load data
    train_ds, val_ds = load_data(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
    NUM_CLASSES = len(train_ds.class_names)
    class_names = train_ds.class_names

    # Preprocess data
    train_ds, val_ds, test_ds = preprocess(train_ds, val_ds, TEST_RATIO)

    # Build model
    model = build_model(IMAGE_SIZE + (3,), NUM_CLASSES, BASE_LEARNING_RATE)
    print(type(model))
    print(model.summary())

    # Train model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.2,
            patience=3,
            verbose=1,
            cooldown=5
        )
    ]
    history = train_model(model, train_ds, val_ds, INITIAL_EPOCHS, callbacks)

    # Fine-tune model
    base_model = model.layers[1]
    history_fine = fine_tune_model(model, base_model, FINE_TUNE_AT, BASE_LEARNING_RATE, INITIAL_EPOCHS + FINE_TUNE_EPOCHS, callbacks)

    # Save model
    model.class_names = class_names
    save_model(model, DATA_DIR.replace("raw", "") + "TL_180px_32b_20e_model.keras")
