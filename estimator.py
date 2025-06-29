import argparse
from pathlib import Path

from sagemaker.estimator import Estimator

from config import Config
from utils import (
    generate_requirements_with_uv,
    rebuild_training_image_if_requirements_changed,
)

# Get configuration
config = Config()

# Get the model version from version.txt
model_version = config.get_model_version()

# Get data paths
data_paths = config.get_data_paths()

if __name__ == "__main__":
    # Get the register_model argument from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the model after training",
    )
    args = parser.parse_args()

    requirements_path = generate_requirements_with_uv(
        Path("pyproject.toml"), Path("uv.lock")
    )

    # Add the register-model argument to hyperparameters
    hyperparameters = config.DEFAULT_HYPERPARAMETERS.copy()
    hyperparameters["register-model"] = args.register_model

    estimator = Estimator(
        image_uri=f"{config.ECR_URI}:{config.MODEL_NAME}",
        role=config.get_sagemaker_role_arn(),
        source_dir=".",
        entry_point="training.py",
        instance_count=1,
        instance_type=config.DEFAULT_TRAINING_INSTANCE_TYPE,
        base_job_name=f"{config.MODEL_NAME}-training",
        hyperparameters=hyperparameters,
        environment={"MLFLOW_TRACKING_URI": config.get_mlflow_tracking_uri()},
        dependencies=[requirements_path],
    )

    rebuild_training_image_if_requirements_changed(
        requirements_path, Path(config.REQUIREMENTS_HASH_FILE), config.BUILD_SCRIPT
    )

    estimator.fit({"train": data_paths["train_data_path"], "test": data_paths["test_data_path"]}, wait=True)
