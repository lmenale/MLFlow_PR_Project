# tensorflow 2.x core api
import tensorflow as tf
import os
from tf.keras.preprocessing.image import ImageDataGenerator

def load_data(data_path):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        data_path,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    return train_generator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    
    # Set a random seed for reproducible results
    tf.random.set_seed(42)

    data_generator = load_data(args.data_path)

    # Load dataset
    # dataset = fetch_california_housing(as_frame=True)["frame"]

    # Further model training here
