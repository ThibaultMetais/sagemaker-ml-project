"""
MLflow inference testing module for the SageMaker + MLflow project.

This module provides functionality to test MLflow model inference locally
before deployment. It can load models from either MLflow runs or the
model registry and perform inference on test data.

The module is useful for validating model performance and ensuring that
models work correctly before deploying them to SageMaker endpoints.
"""

import argparse

import mlflow.sklearn
import pandas as pd

from config import Config


def test_mlflow_inference(run_id: str = None):
    """
    Test MLflow model inference on test data.

    This function loads a trained model from either an MLflow run or the
    model registry and performs inference on test data. It can be used
    to validate model performance and ensure the model works correctly
    before deployment.

    Args:
        run_id (str, optional): MLflow run ID to load the model from.
                               If None, loads the model from the registry
                               using the semantic version tag.
                               Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If the model cannot be loaded or if no model version
                   is found with the specified semantic version tag.

    Note:
        The function expects test data to be available at the configured
        S3 path. The test data should have the same format as the training
        data, with a 'Class' column that will be removed for inference.

        If run_id is provided, the model is loaded directly from the run.
        Otherwise, the function searches the model registry for a version
        with the semantic version tag matching the current project version.
    """
    # Initialize configuration
    config = Config()

    # Get the model version from version.txt for model selection
    model_version = config.get_model_version()

    # Get S3 data paths for test data
    data_paths = config.get_data_paths()

    # Configure MLflow tracking URI
    mlflow.set_tracking_uri(config.get_mlflow_tracking_uri())
    client = mlflow.tracking.MlflowClient()

    # Load model from run or registry
    model = None
    if run_id:
        # Load model directly from a specific MLflow run
        print(f"Loading model from run ID: {run_id}")
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    else:
        # Load model from the MLflow model registry
        print(f"Loading model from registry with semver tag: {model_version}")

        # Search for all versions of the registered model
        versions = client.search_model_versions(f"name='{config.MODEL_NAME}'")

        # Find the version with the matching semantic version tag
        target_version = None
        for version in versions:
            if version.tags.get("semver") == model_version:
                target_version = version
                break

        if not target_version:
            raise ValueError(f"No version of model '{config.MODEL_NAME}' with tag 'semver={model_version}' found.")

        # Load the model from the registry
        model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{target_version.version}")

    # Validate that the model was loaded successfully
    if not model:
        raise ValueError("Model not loaded successfully")

    # Load test data for inference
    print("Loading test data...")
    df = pd.read_csv(data_paths["test_data_path"] + "/dry-bean-test.csv")

    # Remove the target column for inference
    # The model expects only feature columns as input
    df = df.drop("Class", axis=1)

    # Perform inference on the test data
    print("Running predictions...")
    predictions = model.predict(df)

    # Display prediction results
    print(f"Number of predictions: {len(predictions)}")
    print("Predictions:", predictions)

    # Display prediction statistics
    # Get value counts and sort by index (class labels)
    prediction_counts = pd.Series(predictions).value_counts().sort_index()
    print("Prediction distribution:")
    for pred, count in prediction_counts.items():
        print(f"  Class {pred}: {count} samples")


if __name__ == "__main__":
    # Command line interface for testing MLflow inference
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, help="MLflow run ID to load model from (optional)")
    args = parser.parse_args()

    # Execute the inference test
    test_mlflow_inference(args.run_id)
