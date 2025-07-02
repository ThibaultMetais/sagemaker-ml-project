"""
Model training module for the SageMaker + MLflow project.

This module handles the complete model training pipeline, including data loading,
model training, evaluation, and MLflow integration. It supports both local
training and SageMaker training environments, with optional model registration
in the MLflow model registry.

The module trains a Random Forest classifier on the UCI Dry Bean dataset,
evaluates performance using balanced accuracy, and logs experiments and models
to MLflow for tracking and versioning.
"""

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from config import Config

# Initialize configuration
config = Config()

# Model name for MLflow registry
REGISTERED_MODEL_NAME = config.MODEL_NAME

# MLflow tracking URI for experiment logging
MLFLOW_TRACKING_URI = config.get_mlflow_tracking_uri()

# Get the model version from version.txt for model tagging
model_version = config.get_model_version()


def train(train_path, test_path, params, model_dir, register_model: bool = False):
    """
    Train a Random Forest classifier and optionally register it in MLflow.

    This function performs the complete training pipeline:
    1. Loads training and test data
    2. Prepares features and target variables
    3. Trains a Random Forest classifier with specified hyperparameters
    4. Evaluates model performance on training and test sets
    5. Logs experiment metrics and parameters to MLflow
    6. Optionally registers the model in the MLflow model registry
    7. Saves the trained model locally

    Args:
        train_path (str): Path to the training data CSV file.
        test_path (str): Path to the test data CSV file.
        params (dict): Dictionary containing hyperparameters for the model.
                       Expected keys: 'n_estimators', 'min_samples_leaf'.
        model_dir (str): Directory path where the trained model will be saved.
        register_model (bool): If True, registers the model in MLflow model registry.
                              If False, only logs the model without registration.
                              Defaults to False.

    Returns:
        None

    Note:
        The function expects the input CSV files to have a 'Class' column as the target
        variable. All other columns are treated as features.

        The model is saved as 'model.pkl' in the specified model directory.

        If register_model=True, the model is tagged with the semantic version
        from version.txt for easy identification and versioning.
    """
    print("Reading data...")
    # Load training and test datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Building training and testing datasets...")
    # Prepare features (X) and target (y) for both training and test sets
    # Remove the 'Class' column from features, keep it as target
    X_train = train_df.drop("Class", axis=1)
    X_test = test_df.drop("Class", axis=1)
    y_train = train_df["Class"]
    y_test = test_df["Class"]

    print("Training model...")
    # Configure MLflow tracking and experiment settings
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{config.MODEL_NAME}")

    # Initialize and train the Random Forest classifier
    # n_jobs=-1 uses all available CPU cores for parallel training
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        min_samples_leaf=params["min_samples_leaf"],
        n_jobs=-1,  # Use all CPU cores for faster training
    )

    # Fit the model to the training data
    model.fit(X_train, y_train)

    print("Validating model...")
    # Evaluate model performance using balanced accuracy
    # Balanced accuracy is appropriate for imbalanced datasets
    bal_acc_train = balanced_accuracy_score(y_train, model.predict(X_train))
    bal_acc_test = balanced_accuracy_score(y_test, model.predict(X_test))

    print(f"Train balanced accuracy: {bal_acc_train:.3f}")
    print(f"Test balanced accuracy: {bal_acc_test:.3f}")

    # Start MLflow run for experiment tracking
    with mlflow.start_run():
        # Log hyperparameters for experiment tracking
        mlflow.log_params(params)

        # Log performance metrics
        mlflow.log_metrics({"bal_acc_train": bal_acc_train, "bal_acc_test": bal_acc_test})

        # Infer the model signature for later model serving
        # This defines the expected input/output format for the model
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model to MLflow without automatic registration
        # This creates a logged model that can be registered later
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train,  # Provide example input for model serving
        )

        if register_model:
            # Explicitly register the model in the MLflow model registry
            # This creates a versioned model that can be deployed
            registered_model = mlflow.register_model(logged_model.model_uri, REGISTERED_MODEL_NAME)

            # Set a custom semantic version tag for easy identification
            # This links the MLflow model version to the project version
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=registered_model.version,
                key="semver",
                value=model_version,
            )

            print(f"Model registered with version {registered_model.version}, tagged with {model_version}")

    # Save the trained model locally for immediate use
    # This is useful for local testing and debugging
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir + "/model.pkl")
    print(f"Model saved to {model_dir}/model.pkl")


if __name__ == "__main__":
    print("Extracting arguments...")
    # Parse command line arguments for training configuration
    parser = argparse.ArgumentParser()

    # Hyperparameters for the Random Forest classifier
    parser.add_argument("--n-estimators", type=int, default=10, help="Number of trees in the Random Forest")
    parser.add_argument(
        "--min-samples-leaf", type=int, default=3, help="Minimum number of samples required at a leaf node"
    )

    # Data, model, and output directories
    # These are typically set by SageMaker when running in training mode
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"), help="Directory to save the trained model"
    )
    parser.add_argument(
        "--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"), help="Directory containing training data"
    )
    parser.add_argument(
        "--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"), help="Directory containing test data"
    )
    parser.add_argument("--train-file", type=str, default="dry-bean-train.csv", help="Training data filename")
    parser.add_argument("--test-file", type=str, default="dry-bean-test.csv", help="Test data filename")

    # Optional flag to register the model in MLflow Model Registry
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register model in MLflow Model Registry",
    )

    # Parse arguments, ignoring unknown arguments (useful for SageMaker compatibility)
    args, _ = parser.parse_known_args()

    # Extract the register_model flag
    register_model = args.register_model

    # Prepare hyperparameters dictionary
    params = {
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
    }

    # Execute the training pipeline
    train(
        os.path.join(args.train, args.train_file),
        os.path.join(args.test, args.test_file),
        params,
        args.model_dir,
        register_model=register_model,
    )
