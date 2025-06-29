"""
Configuration settings for the AI Roster project.

This module centralizes all configuration values using environment variables
and a .env file to keep sensitive values private and not hardcoded in source code.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for the AI Roster project."""
    
    # AWS Configuration - Required environment variables
    AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID")
    if not AWS_ACCOUNT_ID:
        raise ValueError("AWS_ACCOUNT_ID environment variable is required")
    
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    
    # S3 Configuration - Required environment variables
    S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
    if not S3_BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME environment variable is required")
    
    S3_BUCKET_URI = f"s3://{S3_BUCKET_NAME}"
    DEPLOYMENT_BUCKET_PREFIX = os.getenv("DEPLOYMENT_BUCKET_PREFIX", "sagemaker-models")
    
    # ECR Configuration
    ECR_REPO_NAME = os.getenv("ECR_REPO_NAME", "sagemaker/training_images")
    ECR_URI = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO_NAME}"
    
    # SageMaker Configuration - Required environment variables
    SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
    if not SAGEMAKER_ROLE_ARN:
        raise ValueError("SAGEMAKER_ROLE_ARN environment variable is required")
    
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not MLFLOW_TRACKING_URI:
        raise ValueError("MLFLOW_TRACKING_URI environment variable is required")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "my-example-project")
    
    # Instance Configuration
    DEFAULT_TRAINING_INSTANCE_TYPE = os.getenv("DEFAULT_TRAINING_INSTANCE_TYPE", "ml.c5.xlarge")
    DEFAULT_INFERENCE_INSTANCE_TYPE = os.getenv("DEFAULT_INFERENCE_INSTANCE_TYPE", "ml.m5.large")
    DEFAULT_INSTANCE_COUNT = int(os.getenv("DEFAULT_INSTANCE_COUNT", "1"))
    
    # Hyperparameters
    DEFAULT_HYPERPARAMETERS = {
        "n-estimators": int(os.getenv("N_ESTIMATORS", "100")),
        "min-samples-leaf": int(os.getenv("MIN_SAMPLES_LEAF", "3")),
    }
    
    # File paths
    VERSION_FILE = os.getenv("VERSION_FILE", "version.txt")
    REQUIREMENTS_HASH_FILE = os.getenv("REQUIREMENTS_HASH_FILE", "requirements_hash.txt")
    BUILD_SCRIPT = os.getenv("BUILD_SCRIPT", "build_and_publish.sh")
    
    @classmethod
    def get_model_version(cls) -> str:
        """Get the current model version from version.txt."""
        try:
            with open(cls.VERSION_FILE, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Version file {cls.VERSION_FILE} not found")
    
    @classmethod
    def get_data_paths(cls) -> dict:
        """Get the data paths for the current model version."""
        model_version = cls.get_model_version()
        data_prefix = f"{cls.MODEL_NAME}/{model_version}/data"
        
        return {
            "data_prefix": data_prefix,
            "data_path": f"{cls.S3_BUCKET_URI}/{data_prefix}",
            "train_data_path": f"{cls.S3_BUCKET_URI}/{data_prefix}/train",
            "test_data_path": f"{cls.S3_BUCKET_URI}/{data_prefix}/test",
        }
    
    @classmethod
    def get_deployment_bucket(cls, environment: str) -> str:
        """Get the deployment bucket name for a given environment."""
        return f"{cls.DEPLOYMENT_BUCKET_PREFIX}-{environment}"
    
    @classmethod
    def get_endpoint_name(cls, environment: str) -> str:
        """Get the SageMaker endpoint name for a given environment."""
        model_version = cls.get_model_version()
        return f"{cls.MODEL_NAME}-v{model_version.replace('.', '-')}-{environment}"
    
    @classmethod
    def get_ecr_image_uri(cls, image_tag: Optional[str] = None) -> str:
        """Get the ECR image URI for the model."""
        if image_tag is None:
            image_tag = cls.MODEL_NAME
        return f"{cls.AWS_ACCOUNT_ID}.dkr.ecr.{cls.AWS_REGION}.amazonaws.com/{image_tag}"
    
    @classmethod
    def get_mlflow_tracking_uri(cls) -> str:
        """Get the MLflow tracking URI."""
        return cls.MLFLOW_TRACKING_URI
    
    @classmethod
    def get_sagemaker_role_arn(cls) -> str:
        """Get the SageMaker execution role ARN."""
        return cls.SAGEMAKER_ROLE_ARN


# Environment-specific overrides
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
    """Get the appropriate configuration for the given environment."""
    config_map = {
        "dev": DevConfig,
        "staging": StagingConfig,
        "production": ProductionConfig,
        "prod": ProductionConfig,
    }
    
    return config_map.get(environment.lower(), DevConfig) 