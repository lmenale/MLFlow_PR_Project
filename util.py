from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras import Input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.MobileNetV2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, History, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.Model import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory


def get_ds_from_file(file_path: str, IMAGE_SIZE: tuple, BATCH_SIZE: int) -> Dataset:
    """
    Creates three datasets from the given file path.

    Args:
        file_path (str): The path to the directory containing the image files.
        IMAGE_SIZE (tuple): The desired size of the images in the datasets.
        BATCH_SIZE (int): The number of images to include in each batch.

    Returns:
        tuple: A tuple containing three datasets - train_ds, test_ds, and val_ds.
    """
    # Create the training dataset using 80% of the data
    train_ds = image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Create the testing dataset using 10% of the data
    test_ds = image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        subset="testing",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    # Create the validation dataset using 10% of the data
    val_ds = image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    return train_ds, test_ds, val_ds

def configure_for_performance(ds: Dataset) -> Dataset:
    """
    Configures a dataset for performance by caching and prefetching.

    Args:
        ds (Dataset): The input dataset to be configured.

    Returns:
        Dataset: The configured dataset.

    """
    auto_tune = AUTOTUNE

    return ds.cache().prefetch(buffer_size=auto_tune)

def create_transfer_learning_model(dataset: Dataset, IMAGE_SIZE: tuple, NUM_CLASSES: int) -> Model:
    """
    Creates a transfer learning model using MobileNetV2 as the base model.

    Args:
        dataset (Dataset): The dataset used for training the model.
        IMAGE_SIZE (tuple): The size of the input images.
        NUM_CLASSES (int): The number of classes in the dataset.

    Returns:
        Model: The transfer learning model.

    """
    # This model expects pixel values in [-1, 1], but at this point, the pixel values in your images are in [0, 255]
    base_model = MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                            include_top=False,
                            weights='imagenet')

    image_batch, label_batch = next(iter(dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    # Freeze the base model
    base_model.trainable = False

    # Let's take a look at the base model architecture
    base_model.summary()

    # Add a classification head
    global_average_layer = GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    prediction_layer = Dense(NUM_CLASSES, activation='softmax')

    # Build the model
    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    layer = preprocess_input(inputs)
    layer = base_model(layer, training=False)
    layer = global_average_layer(layer)
    layer = Dropout(0.2)(layer)
    outputs = prediction_layer(layer)

    model = Model(inputs, outputs)

    return model

def compile_transfer_learning_model(model: Model, learning_rate: float) -> None:
    """
    Compiles a transfer learning model with the specified learning rate.

    Args:
        model (Model): The transfer learning model to compile.
        learning_rate (float): The learning rate to use for training.

    Returns:
        None
    """
    base_learning_rate = learning_rate

    model.compile(optimizer=Adam(learning_rate=base_learning_rate),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

def train_transfer_learning_model(model: Model, train_ds: Dataset, val_ds: Dataset, epochs: int) -> History:
    tl_epochs = epochs

    loss0, accuracy0 = model.evaluate(val_ds)
    model.fit(train_ds, validation_data=val_ds, epochs=tl_epochs)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 10, verbose = 1, restore_best_weights = True)
    lr_plateau = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.2, patience = 3, verbose = 1, cooldown = 5)

    history = model.fit(train_ds,
                        epochs=tl_epochs,
                        validation_data=val_ds,
                        callbacks = [early_stopping, lr_plateau])

    return history

def tuning_transfer_learning_model(model: Model, train_ds: Dataset, val_ds: Dataset, epochs: int) -> None:
    model.trainable = True

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(model.layers))

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:epochs]:
        layer.trainable = False

    # Compile the model
    compile_transfer_learning_model(model, 0.0001)

    # Fine-tune from this layer onwards
    fine_tune_epochs = 10 + epochs
    train_transfer_learning_model(model, train_ds, val_ds, fine_tune_epochs)
