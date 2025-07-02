"""
SageMaker estimator configuration module.

This module provides functionality to create and configure SageMaker estimators
for model training. It handles dependency management, Docker image rebuilding,
and estimator configuration with proper hyperparameters and environment variables.

The module integrates with the project's uv-based dependency management system
and ensures that training images are rebuilt only when dependencies change.
"""

import argparse
from pathlib import Path

from sagemaker.estimator import Estimator

from config import Config
from utils import (
    generate_requirements_with_uv,
    rebuild_training_image_if_requirements_changed,
)

# Initialize configuration
config = Config()

# Get the model version from version.txt for image tagging
model_version = config.get_model_version()

# Get S3 data paths for training data
data_paths = config.get_data_paths()


def build_estimator(register_model: bool = False) -> Estimator:
    """
    Build and configure a SageMaker estimator for model training.

    This function creates a SageMaker estimator with the following features:
    1. Generates requirements.txt from pyproject.toml using uv
    2. Checks if dependencies have changed and rebuilds Docker image if necessary
    3. Configures the estimator with proper hyperparameters and environment variables
    4. Sets up data channels and model output directory
    5. Configures the training entry point and source code

    Args:
        register_model (bool): If True, adds the register-model flag to hyperparameters
                              to register the model in MLflow after training.
                              Defaults to False.

    Returns:
        Estimator: A configured SageMaker estimator ready for training.

    Note:
        The function automatically handles Docker image rebuilding when dependencies
        change, ensuring that the training environment is always up-to-date.

        The estimator is configured to use the project's ECR repository and
        includes all necessary environment variables for MLflow integration.
    """
    # Generate requirements.txt from pyproject.toml using uv
    # This ensures the Docker image has all necessary dependencies
    requirements_path = generate_requirements_with_uv(
        Path(__file__).parent.parent / "pyproject.toml",
        Path(__file__).parent.parent / "uv.lock",
    )

    # Get the version file path for inclusion in the training job
    version_file = Path(__file__).parent.parent / config.VERSION_FILE

    # Prepare hyperparameters for the training job
    # Copy default hyperparameters and add the register-model flag if requested
    hyperparameters = config.DEFAULT_HYPERPARAMETERS.copy()
    hyperparameters["register-model"] = register_model

    # Check if requirements have changed and rebuild Docker image if necessary
    # This ensures the training environment is always up-to-date
    rebuild_training_image_if_requirements_changed(
        requirements_path,
        Path(__file__).parent.parent / config.REQUIREMENTS_HASH_FILE,
        Path(__file__).parent.parent / config.BUILD_SCRIPT,
    )

    # Create and configure the SageMaker estimator
    estimator = Estimator(
        # ECR image URI for the training container
        image_uri=f"{config.ECR_URI}/{config.ECR_TRAINING_REPO_NAME}:{config.MODEL_NAME}",
        # SageMaker execution role for AWS permissions
        role=config.get_sagemaker_role_arn(),
        # Source code directory and entry point
        source_dir="src",  # Directory containing training code
        entry_point="training.py",  # Training script to execute
        # Training instance configuration
        instance_count=1,  # Number of training instances
        instance_type=config.DEFAULT_TRAINING_INSTANCE_TYPE,  # Instance type (e.g., ml.c5.xlarge)
        # Job naming and hyperparameters
        base_job_name=f"{config.MODEL_NAME}-training",  # Base name for training jobs
        hyperparameters=hyperparameters,  # Model hyperparameters
        # Environment variables for the training container
        environment={
            "AWS_REGION": config.AWS_REGION,  # AWS region for services
            "MLFLOW_TRACKING_URI": config.get_mlflow_tracking_uri(),  # MLflow tracking server
            "MODEL_NAME": config.MODEL_NAME,  # Model name for registry
            "S3_BUCKET_NAME": config.S3_BUCKET_NAME,  # S3 bucket for data
            "N_ESTIMATORS": str(config.DEFAULT_HYPERPARAMETERS["n-estimators"]),  # Number of trees
            "MIN_SAMPLES_LEAF": str(config.DEFAULT_HYPERPARAMETERS["min-samples-leaf"]),  # Min samples per leaf
            "VERSION_FILE": config.VERSION_FILE,  # Version file path
            "REQUIREMENTS_HASH_FILE": config.REQUIREMENTS_HASH_FILE,  # Requirements hash file path
        },
        # Additional files to include in the training job
        dependencies=[requirements_path, version_file],  # Requirements and version files
    )

    return estimator


if __name__ == "__main__":
    # Command line interface for running training jobs
    # Get the register_model argument from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the model in MLflow after training",
    )
    args = parser.parse_args()

    # Build the estimator with the specified configuration
    estimator = build_estimator(args.register_model)

    # Start the training job with data channels
    # The training job will use data from S3 and save the model to the model directory
    estimator.fit(
        {
            "train": data_paths["train_data_path"],  # Training data channel
            "test": data_paths["test_data_path"],  # Test data channel
        },
        wait=True,  # Wait for the training job to complete
    )
