import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.preprocessing.image import load_img, img_to_array
import os
import zipfile

def load_data(metadata_path, images_dir):
    df = pd.read_csv(metadata_path)
    images = []
    for image in df['Image_name']:
        image_path = os.path.join(images_dir, image)
        image = load_img(image_path, target_size=(128, 128))
        image = img_to_array(image)
        images.append(image)
    images = np.array(images, dtype='float32') / 255.0
    return images, df
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_dataset(metadata_path, images_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        # Add metadata
        zipf.write(metadata_path, os.path.basename(metadata_path))
        # Add images
        for root, _, files in os.walk(images_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), images_dir))

if __name__ == "__main__":
    mlflow.start_run()
    
    metadata_path = 'data/metadata.csv'
    images_dir = 'data/images'
    
    images, df = load_data(metadata_path, images_dir)
    labels = df['Healthy'].values
    
    model = create_model((128, 128, 3))
    model.fit(images, labels, epochs=10, batch_size=2)
    
    # Save the model
    model_path = "model.h5"
    model.save(model_path)
    
    # Log the model
    mlflow.keras.log_model(model, "model")
    mlflow.log_artifact(model_path)
    
    # Save and log the dataset
    dataset_zip = "dataset.zip"
    save_dataset(metadata_path, images_dir, dataset_zip)
    mlflow.log_artifact(dataset_zip)
    
    mlflow.end_run()
    
    # Load the model to verify
    loaded_model = load_model(model_path)
    loaded_model.summary()
    
    predictions = loaded_model.predict(images)
    print(predictions)
