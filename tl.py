import tensorflow as tf
import os

from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

BATCH_SIZE = 32
IMAGE_SIZE = (180, 180)
data_dir = "../data/raw"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(class_names)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

print("Number of validation batches: %d" % tf.data.experimental.cardinality(val_ds))
print("Number of test batches: %d" % tf.data.experimental.cardinality(test_ds))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# This model expects pixel values in [-1, 1], but at this point, the pixel values in your images are in [0, 255]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = MobileNetV2(
    input_shape=(180, 180, 3), include_top=False, weights="imagenet"
)

image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# Freeze the base model
base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

# Add a classification head
global_average_layer = layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
prediction_layer = layers.Dense(len(class_names), activation="softmax")

# Build the model
inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
layer = preprocess_input(inputs)
layer = base_model(layer, training=False)
layer = global_average_layer(layer)
layer = tf.keras.layers.Dropout(0.2)(layer)
outputs = prediction_layer(layer)

model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

initial_epochs = 10

loss0, accuracy0 = model.evaluate(val_ds)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True
)
lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.2, patience=3, verbose=1, cooldown=5
)

history = model.fit(
    train_ds,
    epochs=initial_epochs,
    validation_data=val_ds,
    callbacks=[early_stopping, lr_plateau],
)

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=len(history.epoch),
    validation_data=val_ds,
    callbacks=[early_stopping, lr_plateau],
)

# Saving the model
model.save(data_dir + "/TL_180px_32b_20e_model.keras")
