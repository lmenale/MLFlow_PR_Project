# MLFlow_PR_Project
This is the MLFlow project to keep track of our datasets.

## Running Your MLFlow Project
To run your MLFlow project, you can use the following command from the terminal. This ensures that the MLFlow runs within the context defined by your MLproject and requirements.txt:

```bash
mlflow run MyMLFlowProject -P model_type=gradient_boosting
```

## Set Up an MLFlow Tracking Server
If you're working in a team, setting up an MLFlow tracking server is useful. Run the following command in a dedicated terminal:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0
```

This command starts an MLFlow server with SQLite as the backend for storing experiments and a local directory for storing artifacts.

## Accessing the MLFlow UI
Access the MLFlow Tracking UI by navigating to http://localhost:5000 on your web browser. This UI can be used by your team to view experiments, compare runs, and manage models.