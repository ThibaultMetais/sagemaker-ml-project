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


def build_estimator(register_model: bool = False) -> Estimator:
    requirements_path = generate_requirements_with_uv(
        Path(__file__).parent.parent / "pyproject.toml",
        Path(__file__).parent.parent / "uv.lock",
    )
    version_file = Path(__file__).parent.parent / config.VERSION_FILE

    # Add the register-model argument to hyperparameters
    hyperparameters = config.DEFAULT_HYPERPARAMETERS.copy()
    hyperparameters["register-model"] = register_model

    rebuild_training_image_if_requirements_changed(
        requirements_path,
        Path(__file__).parent.parent / config.REQUIREMENTS_HASH_FILE,
        Path(__file__).parent.parent / config.BUILD_SCRIPT,
    )

    estimator = Estimator(
        image_uri=f"{config.ECR_URI}/{config.ECR_TRAINING_REPO_NAME}:{config.MODEL_NAME}",
        role=config.get_sagemaker_role_arn(),
        source_dir="src",
        entry_point="training.py",
        instance_count=1,
        instance_type=config.DEFAULT_TRAINING_INSTANCE_TYPE,
        base_job_name=f"{config.MODEL_NAME}-training",
        hyperparameters=hyperparameters,
        environment={
            "AWS_REGION": config.AWS_REGION,
            "MLFLOW_TRACKING_URI": config.get_mlflow_tracking_uri(),
            "MODEL_NAME": config.MODEL_NAME,
            "S3_BUCKET_NAME": config.S3_BUCKET_NAME,
            "N_ESTIMATORS": str(config.DEFAULT_HYPERPARAMETERS["n-estimators"]),
            "MIN_SAMPLES_LEAF": str(config.DEFAULT_HYPERPARAMETERS["min-samples-leaf"]),
            "VERSION_FILE": config.VERSION_FILE,
            "REQUIREMENTS_HASH_FILE": config.REQUIREMENTS_HASH_FILE,
        },
        dependencies=[requirements_path, version_file],
    )

    return estimator


if __name__ == "__main__":
    # Get the register_model argument from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the model after training",
    )
    args = parser.parse_args()

    estimator = build_estimator(args.register_model)
    estimator.fit(
        {"train": data_paths["train_data_path"], "test": data_paths["test_data_path"]},
        wait=True,
    )
