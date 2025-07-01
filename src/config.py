"""
Centralized configuration for the project.
This version is "lazy": no environment variables are resolved at import time,
but only when they are used (via properties or methods).
This prevents crashes in partial environments (CI, container, local, etc.).
"""

import os
from typing import Optional

import boto3
from dotenv import load_dotenv

# Chargement du .env local si présent
load_dotenv()


def _get_env_or_raise(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(f"{name} environment variable is required")
    return val


class Config:
    """Configuration centralisée pour le projet."""

    @property
    def AWS_ACCOUNT_ID(self) -> str:
        return boto3.client("sts").get_caller_identity()["Account"]

    @property
    def AWS_REGION(self) -> str:
        return os.getenv("AWS_REGION", "us-east-1")

    @property
    def S3_BUCKET_NAME(self) -> str:
        return _get_env_or_raise("S3_BUCKET_NAME")

    @property
    def S3_BUCKET_URI(self) -> str:
        return f"s3://{self.S3_BUCKET_NAME}"

    @property
    def DEPLOYMENT_BUCKET_PREFIX(self) -> str:
        return os.getenv("DEPLOYMENT_BUCKET_PREFIX", "sagemaker-models")

    @property
    def ECR_TRAINING_REPO_NAME(self) -> str:
        return os.getenv("ECR_TRAINING_REPO_NAME", "sagemaker/training_images")

    @property
    def ECR_URI(self) -> str:
        return f"{self.AWS_ACCOUNT_ID}.dkr.ecr.{self.AWS_REGION}.amazonaws.com"

    @property
    def SAGEMAKER_ROLE_ARN(self) -> str:
        return _get_env_or_raise("SAGEMAKER_ROLE_ARN")

    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        return _get_env_or_raise("MLFLOW_TRACKING_URI")

    @property
    def MODEL_NAME(self) -> str:
        return os.getenv("MODEL_NAME", "my-example-project")

    @property
    def DEFAULT_TRAINING_INSTANCE_TYPE(self) -> str:
        return os.getenv("DEFAULT_TRAINING_INSTANCE_TYPE", "ml.c5.xlarge")

    @property
    def DEFAULT_INFERENCE_INSTANCE_TYPE(self) -> str:
        return os.getenv("DEFAULT_INFERENCE_INSTANCE_TYPE", "ml.m5.large")

    @property
    def DEFAULT_INSTANCE_COUNT(self) -> int:
        return int(os.getenv("DEFAULT_INSTANCE_COUNT", "1"))

    @property
    def DEFAULT_HYPERPARAMETERS(self) -> dict:
        return {
            "n-estimators": int(os.getenv("N_ESTIMATORS", "100")),
            "min-samples-leaf": int(os.getenv("MIN_SAMPLES_LEAF", "3")),
        }

    @property
    def VERSION_FILE(self) -> str:
        return os.getenv("VERSION_FILE", "version.txt")

    @property
    def REQUIREMENTS_HASH_FILE(self) -> str:
        return os.getenv("REQUIREMENTS_HASH_FILE", "requirements_hash.txt")

    @property
    def BUILD_SCRIPT(self) -> str:
        return os.getenv("BUILD_SCRIPT", "build_and_publish.sh")

    def get_model_version(self) -> str:
        try:
            with open(self.VERSION_FILE) as f:
                return f.read().strip()
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Version file {self.VERSION_FILE} not found") from err

    def get_data_paths(self) -> dict:
        model_version = self.get_model_version()
        prefix = f"{self.MODEL_NAME}/{model_version}/data"
        return {
            "data_prefix": prefix,
            "data_path": f"{self.S3_BUCKET_URI}/{prefix}",
            "train_data_path": f"{self.S3_BUCKET_URI}/{prefix}/train",
            "test_data_path": f"{self.S3_BUCKET_URI}/{prefix}/test",
        }

    def get_deployment_bucket(self, env: str) -> str:
        return f"{self.DEPLOYMENT_BUCKET_PREFIX}-{env}"

    def get_endpoint_name(self, env: str) -> str:
        version = self.get_model_version().replace(".", "-")
        return f"{self.MODEL_NAME}-v{version}-{env}"

    def get_ecr_image_uri(self, tag: Optional[str] = None) -> str:
        tag = tag or self.MODEL_NAME
        return f"{self.ECR_URI}/{tag}"

    def get_mlflow_tracking_uri(self) -> str:
        return self.MLFLOW_TRACKING_URI

    def get_sagemaker_role_arn(self) -> str:
        return self.SAGEMAKER_ROLE_ARN


# Configuration par environnement (peu utile ici, mais compatible)
class DevConfig(Config):
    pass


class StagingConfig(Config):
    pass


class ProductionConfig(Config):
    pass


def get_config(environment: str = "dev") -> Config:
    return {
        "dev": DevConfig,
        "staging": StagingConfig,
        "production": ProductionConfig,
        "prod": ProductionConfig,
    }.get(environment.lower(), DevConfig)()
