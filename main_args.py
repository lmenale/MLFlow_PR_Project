import mlflow
import sys
import argparse

def train_model(data_file, model_type):
    with mlflow.start_run():
        # Your training code here
        mlflow.log_param("data_file", data_file)
        mlflow.log_param("model_type", model_type)
        # log metrics
        mlflow.log_metric("accuracy", 0.95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLFlow example')
    parser.add_argument('--data_file', type=str, default='data/data.csv')
    parser.add_argument('--model_type', type=str, default='random_forest')
    args = parser.parse_args()
    
    train_model(args.data_file, args.model_type)
