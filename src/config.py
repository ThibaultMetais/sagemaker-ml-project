"""
Centralized configuration for the project.

This module provides a "lazy" configuration system where environment variables
are resolved only when accessed, not at import time. This prevents crashes
in partial environments (CI, container, local, etc.) and allows for flexible
deployment across different environments.

The configuration supports multiple environments (dev, staging, production)
and provides convenient access to AWS resources, MLflow settings, and
project-specific configurations.
"""

import os
from typing import Optional

import boto3
from dotenv import load_dotenv

# Load local .env file if present for development convenience
load_dotenv()


def _get_env_or_raise(name: str) -> str:
    """
    Get an environment variable value or raise an error if not found.

    This is a helper function to ensure required environment variables
    are properly configured before the application runs.

    Args:
        name (str): The name of the environment variable to retrieve.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set or is empty.
    """
    val = os.getenv(name)
    if not val:
        raise ValueError(f"{name} environment variable is required")
    return val


class Config:
    """
    Centralized configuration class for the SageMaker + MLflow project.

    This class provides lazy loading of configuration values from environment
    variables and AWS services. Properties are used to ensure values are
    retrieved only when needed, preventing import-time failures.

    Attributes:
        Various properties that provide access to AWS, MLflow, and project
        configuration values.
    """

    @property
    def AWS_ACCOUNT_ID(self) -> str:
        """
        Get the current AWS account ID from STS.

        Returns:
            str: The AWS account ID of the currently authenticated user/role.
        """
        return boto3.client("sts").get_caller_identity()["Account"]

    @property
    def AWS_REGION(self) -> str:
        """
        Get the AWS region for resource deployment.

        Returns:
            str: The AWS region, defaults to 'us-east-1' if not specified.
        """
        return os.getenv("AWS_REGION", "us-east-1")

    @property
    def S3_BUCKET_NAME(self) -> str:
        """
        Get the S3 bucket name for data and model storage.

        Returns:
            str: The S3 bucket name.

        Raises:
            ValueError: If S3_BUCKET_NAME environment variable is not set.
        """
        return _get_env_or_raise("S3_BUCKET_NAME")

    @property
    def S3_BUCKET_URI(self) -> str:
        """
        Get the full S3 bucket URI.

        Returns:
            str: The S3 bucket URI in the format 's3://bucket-name'.
        """
        return f"s3://{self.S3_BUCKET_NAME}"

    @property
    def DEPLOYMENT_BUCKET_PREFIX(self) -> str:
        """
        Get the prefix for deployment S3 buckets.

        Returns:
            str: The deployment bucket prefix, defaults to 'sagemaker-models'.
        """
        return os.getenv("DEPLOYMENT_BUCKET_PREFIX", "sagemaker-models")

    @property
    def ECR_TRAINING_REPO_NAME(self) -> str:
        """
        Get the ECR repository name for training images.

        Returns:
            str: The ECR repository name, defaults to 'sagemaker/training_images'.
        """
        return os.getenv("ECR_TRAINING_REPO_NAME", "sagemaker/training_images")

    @property
    def ECR_URI(self) -> str:
        """
        Get the ECR registry URI for the current AWS account and region.

        Returns:
            str: The ECR registry URI in the format 'account.dkr.ecr.region.amazonaws.com'.
        """
        return f"{self.AWS_ACCOUNT_ID}.dkr.ecr.{self.AWS_REGION}.amazonaws.com"

    @property
    def SAGEMAKER_ROLE_ARN(self) -> str:
        """
        Get the SageMaker execution role ARN.

        Returns:
            str: The SageMaker execution role ARN.

        Raises:
            ValueError: If SAGEMAKER_ROLE_ARN environment variable is not set.
        """
        return _get_env_or_raise("SAGEMAKER_ROLE_ARN")

    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        """
        Get the MLflow tracking server URI.

        Returns:
            str: The MLflow tracking server URI.

        Raises:
            ValueError: If MLFLOW_TRACKING_URI environment variable is not set.
        """
        return _get_env_or_raise("MLFLOW_TRACKING_URI")

    @property
    def MODEL_NAME(self) -> str:
        """
        Get the model name for registry and deployment.

        Returns:
            str: The model name, defaults to 'my-example-project'.
        """
        return os.getenv("MODEL_NAME", "my-example-project")

    @property
    def DEFAULT_TRAINING_INSTANCE_TYPE(self) -> str:
        """
        Get the default SageMaker training instance type.

        Returns:
            str: The training instance type, defaults to 'ml.c5.xlarge'.
        """
        return os.getenv("DEFAULT_TRAINING_INSTANCE_TYPE", "ml.c5.xlarge")

    @property
    def DEFAULT_INFERENCE_INSTANCE_TYPE(self) -> str:
        """
        Get the default SageMaker inference instance type.

        Returns:
            str: The inference instance type, defaults to 'ml.m5.large'.
        """
        return os.getenv("DEFAULT_INFERENCE_INSTANCE_TYPE", "ml.m5.large")

    @property
    def DEFAULT_INSTANCE_COUNT(self) -> int:
        """
        Get the default number of inference instances.

        Returns:
            int: The number of instances, defaults to 1.
        """
        return int(os.getenv("DEFAULT_INSTANCE_COUNT", "1"))

    @property
    def DEFAULT_HYPERPARAMETERS(self) -> dict:
        """
        Get the default hyperparameters for model training.

        Returns:
            dict: Dictionary containing default hyperparameter values.
        """
        return {
            "n-estimators": int(os.getenv("N_ESTIMATORS", "100")),
            "min-samples-leaf": int(os.getenv("MIN_SAMPLES_LEAF", "3")),
        }

    @property
    def VERSION_FILE(self) -> str:
        """
        Get the path to the version file.

        Returns:
            str: The version file path, defaults to 'version.txt'.
        """
        return os.getenv("VERSION_FILE", "version.txt")

    @property
    def REQUIREMENTS_HASH_FILE(self) -> str:
        """
        Get the path to the requirements hash file.

        Returns:
            str: The requirements hash file path, defaults to 'requirements_hash.txt'.
        """
        return os.getenv("REQUIREMENTS_HASH_FILE", "requirements_hash.txt")

    @property
    def BUILD_SCRIPT(self) -> str:
        """
        Get the path to the build script.

        Returns:
            str: The build script path, defaults to 'build_and_publish.sh'.
        """
        return os.getenv("BUILD_SCRIPT", "build_and_publish.sh")

    def get_model_version(self) -> str:
        """
        Read the model version from the version file.

        Returns:
            str: The model version string.

        Raises:
            FileNotFoundError: If the version file doesn't exist.
        """
        try:
            with open(self.VERSION_FILE) as f:
                return f.read().strip()
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Version file {self.VERSION_FILE} not found") from err

    def get_data_paths(self) -> dict:
        """
        Generate S3 paths for data storage based on model version.

        Returns:
            dict: Dictionary containing various S3 data paths including:
                - data_prefix: Base prefix for data
                - data_path: Full S3 URI for data directory
                - train_data_path: S3 URI for training data
                - test_data_path: S3 URI for test data
        """
        model_version = self.get_model_version()
        prefix = f"{self.MODEL_NAME}/{model_version}/data"
        return {
            "data_prefix": prefix,
            "data_path": f"{self.S3_BUCKET_URI}/{prefix}",
            "train_data_path": f"{self.S3_BUCKET_URI}/{prefix}/train",
            "test_data_path": f"{self.S3_BUCKET_URI}/{prefix}/test",
        }

    def get_deployment_bucket(self, env: str) -> str:
        """
        Get the deployment bucket name for a specific environment.

        Args:
            env (str): The environment name (e.g., 'dev', 'staging', 'prod').

        Returns:
            str: The deployment bucket name in the format 'prefix-env'.
        """
        return f"{self.DEPLOYMENT_BUCKET_PREFIX}-{env}"

    def get_endpoint_name(self, env: str) -> str:
        """
        Generate a SageMaker endpoint name for a specific environment.

        Args:
            env (str): The environment name (e.g., 'dev', 'staging', 'prod').

        Returns:
            str: The endpoint name in the format 'model-name-vversion-env'.
        """
        version = self.get_model_version().replace(".", "-")
        return f"{self.MODEL_NAME}-v{version}-{env}"

    def get_ecr_image_uri(self, tag: Optional[str] = None) -> str:
        """
        Generate the ECR image URI for a given tag.

        Args:
            tag (Optional[str]): The image tag. If None, uses MODEL_NAME.

        Returns:
            str: The full ECR image URI.
        """
        tag = tag or self.MODEL_NAME
        return f"{self.ECR_URI}/{tag}"

    def get_mlflow_tracking_uri(self) -> str:
        """
        Get the MLflow tracking URI.

        Returns:
            str: The MLflow tracking server URI.
        """
        return self.MLFLOW_TRACKING_URI

    def get_sagemaker_role_arn(self) -> str:
        """
        Get the SageMaker execution role ARN.

        Returns:
            str: The SageMaker execution role ARN.
        """
        return self.SAGEMAKER_ROLE_ARN


# Environment-specific configuration classes
# These classes provide a foundation for environment-specific overrides
# if needed in the future, while maintaining compatibility with the base Config.


class DevConfig(Config):
    """Development environment configuration."""

    pass


class StagingConfig(Config):
    """Staging environment configuration."""

    pass


class ProductionConfig(Config):
    """Production environment configuration."""

    pass


def get_config(environment: str = "dev") -> Config:
    """
    Get a configuration instance for the specified environment.

    Args:
        environment (str): The environment name ('dev', 'staging', 'production', or 'prod').

    Returns:
        Config: A configuration instance for the specified environment.

    Note:
        Currently all environments use the same configuration, but this structure
        allows for environment-specific overrides in the future.
    """
    return {
        "dev": DevConfig,
        "staging": StagingConfig,
        "production": ProductionConfig,
        "prod": ProductionConfig,
    }.get(environment.lower(), DevConfig)()
