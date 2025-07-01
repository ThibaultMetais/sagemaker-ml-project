"""
Unit tests for the config.py module.

This module contains comprehensive tests for the Config class and related functionality,
including environment variable handling, AWS service interactions, and configuration
management across different environments.
"""

import os
from unittest.mock import patch

import pytest

from config import Config, DevConfig, ProductionConfig, StagingConfig, _get_env_or_raise, get_config


class TestGetEnvOrRaise:
    """Test cases for the _get_env_or_raise helper function."""

    def test_get_env_or_raise_with_valid_env_var(self):
        """Test that _get_env_or_raise returns the environment variable value when it exists."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = _get_env_or_raise("TEST_VAR")
            assert result == "test_value"

    def test_get_env_or_raise_with_missing_env_var(self):
        """Test that _get_env_or_raise raises ValueError when environment variable is missing."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ValueError, match="TEST_VAR environment variable is required"
        ):
            _get_env_or_raise("TEST_VAR")

    def test_get_env_or_raise_with_empty_env_var(self):
        """Test that _get_env_or_raise raises ValueError when environment variable is empty."""
        with patch.dict(os.environ, {"TEST_VAR": ""}), pytest.raises(
            ValueError, match="TEST_VAR environment variable is required"
        ):
            _get_env_or_raise("TEST_VAR")


class TestConfig:
    """Test cases for the main Config class."""

    def test_aws_account_id_property(self, mock_aws_services):
        """Test that AWS_ACCOUNT_ID property returns the correct account ID."""
        config = Config()
        assert config.AWS_ACCOUNT_ID == "123456789012"

    def test_aws_region_property_with_env_var(self):
        """Test that AWS_REGION property returns environment variable value when set."""
        with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}):
            config = Config()
            assert config.AWS_REGION == "us-west-2"

    def test_aws_region_property_without_env_var(self):
        """Test that AWS_REGION property returns default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.AWS_REGION == "us-east-1"

    def test_s3_bucket_name_property_with_env_var(self):
        """Test that S3_BUCKET_NAME property returns environment variable value."""
        with patch.dict(os.environ, {"S3_BUCKET_NAME": "my-test-bucket"}):
            config = Config()
            assert config.S3_BUCKET_NAME == "my-test-bucket"

    def test_s3_bucket_name_property_without_env_var(self):
        """Test that S3_BUCKET_NAME property raises ValueError when environment variable is missing."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError, match="S3_BUCKET_NAME environment variable is required"):
                _ = config.S3_BUCKET_NAME

    def test_s3_bucket_uri_property(self):
        """Test that S3_BUCKET_URI property constructs correct S3 URI."""
        with patch.dict(os.environ, {"S3_BUCKET_NAME": "my-test-bucket"}):
            config = Config()
            assert config.S3_BUCKET_URI == "s3://my-test-bucket"

    def test_deployment_bucket_prefix_property_with_env_var(self):
        """Test that DEPLOYMENT_BUCKET_PREFIX property returns environment variable value when set."""
        with patch.dict(os.environ, {"DEPLOYMENT_BUCKET_PREFIX": "custom-prefix"}):
            config = Config()
            assert config.DEPLOYMENT_BUCKET_PREFIX == "custom-prefix"

    def test_deployment_bucket_prefix_property_without_env_var(self):
        """Test that DEPLOYMENT_BUCKET_PREFIX property returns default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.DEPLOYMENT_BUCKET_PREFIX == "sagemaker-models"

    def test_ecr_training_repo_name_property_with_env_var(self):
        """Test that ECR_TRAINING_REPO_NAME property returns environment variable value when set."""
        with patch.dict(os.environ, {"ECR_TRAINING_REPO_NAME": "custom/repo"}):
            config = Config()
            assert config.ECR_TRAINING_REPO_NAME == "custom/repo"

    def test_ecr_training_repo_name_property_without_env_var(self):
        """Test that ECR_TRAINING_REPO_NAME property returns default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.ECR_TRAINING_REPO_NAME == "sagemaker/training_images"

    def test_ecr_uri_property(self, mock_aws_services):
        """Test that ECR_URI property constructs correct ECR URI."""
        config = Config()
        expected_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
        assert expected_uri == config.ECR_URI

    def test_sagemaker_role_arn_property_with_env_var(self):
        """Test that SAGEMAKER_ROLE_ARN property returns environment variable value."""
        with patch.dict(os.environ, {"SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789012:role/test-role"}):
            config = Config()
            assert config.SAGEMAKER_ROLE_ARN == "arn:aws:iam::123456789012:role/test-role"

    def test_sagemaker_role_arn_property_without_env_var(self):
        """Test that SAGEMAKER_ROLE_ARN property raises ValueError when environment variable is missing."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError, match="SAGEMAKER_ROLE_ARN environment variable is required"):
                _ = config.SAGEMAKER_ROLE_ARN

    def test_mlflow_tracking_uri_property_with_env_var(self):
        """Test that MLFLOW_TRACKING_URI property returns environment variable value."""
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
            config = Config()
            assert config.MLFLOW_TRACKING_URI == "http://localhost:5000"

    def test_mlflow_tracking_uri_property_without_env_var(self):
        """Test that MLFLOW_TRACKING_URI property raises ValueError when environment variable is missing."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI environment variable is required"):
                _ = config.MLFLOW_TRACKING_URI

    def test_model_name_property_with_env_var(self):
        """Test that MODEL_NAME property returns environment variable value when set."""
        with patch.dict(os.environ, {"MODEL_NAME": "custom-model"}):
            config = Config()
            assert config.MODEL_NAME == "custom-model"

    def test_model_name_property_without_env_var(self):
        """Test that MODEL_NAME property returns default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.MODEL_NAME == "my-example-project"

    def test_default_training_instance_type_property_with_env_var(self):
        """Test that DEFAULT_TRAINING_INSTANCE_TYPE property returns environment variable value when set."""
        with patch.dict(os.environ, {"DEFAULT_TRAINING_INSTANCE_TYPE": "ml.p3.2xlarge"}):
            config = Config()
            assert config.DEFAULT_TRAINING_INSTANCE_TYPE == "ml.p3.2xlarge"

    def test_default_training_instance_type_property_without_env_var(self):
        """Test that DEFAULT_TRAINING_INSTANCE_TYPE property returns default value when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.DEFAULT_TRAINING_INSTANCE_TYPE == "ml.c5.xlarge"

    def test_default_inference_instance_type_property_with_env_var(self):
        """Test that DEFAULT_INFERENCE_INSTANCE_TYPE property returns environment variable value when set."""
        with patch.dict(os.environ, {"DEFAULT_INFERENCE_INSTANCE_TYPE": "ml.c5.2xlarge"}):
            config = Config()
            assert config.DEFAULT_INFERENCE_INSTANCE_TYPE == "ml.c5.2xlarge"

    def test_default_inference_instance_type_property_without_env_var(self):
        """Test that DEFAULT_INFERENCE_INSTANCE_TYPE property returns default value when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.DEFAULT_INFERENCE_INSTANCE_TYPE == "ml.m5.large"

    def test_default_instance_count_property_with_env_var(self):
        """Test that DEFAULT_INSTANCE_COUNT property returns environment variable value when set."""
        with patch.dict(os.environ, {"DEFAULT_INSTANCE_COUNT": "3"}):
            config = Config()
            assert config.DEFAULT_INSTANCE_COUNT == 3

    def test_default_instance_count_property_without_env_var(self):
        """Test that DEFAULT_INSTANCE_COUNT property returns default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.DEFAULT_INSTANCE_COUNT == 1

    def test_default_instance_count_property_with_invalid_env_var(self):
        """Test that DEFAULT_INSTANCE_COUNT property handles invalid environment variable values."""
        with patch.dict(os.environ, {"DEFAULT_INSTANCE_COUNT": "invalid"}), pytest.raises(ValueError):
            config = Config()
            _ = config.DEFAULT_INSTANCE_COUNT

    def test_default_hyperparameters_property_with_env_vars(self):
        """Test that DEFAULT_HYPERPARAMETERS property returns correct values when environment variables are set."""
        with patch.dict(os.environ, {"N_ESTIMATORS": "200", "MIN_SAMPLES_LEAF": "5"}):
            config = Config()
            expected = {"n-estimators": 200, "min-samples-leaf": 5}
            assert expected == config.DEFAULT_HYPERPARAMETERS

    def test_default_hyperparameters_property_without_env_vars(self):
        """Test that DEFAULT_HYPERPARAMETERS property returns default values when environment variables are not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            expected = {"n-estimators": 100, "min-samples-leaf": 3}
            assert expected == config.DEFAULT_HYPERPARAMETERS

    def test_version_file_property_with_env_var(self):
        """Test that VERSION_FILE property returns environment variable value when set."""
        with patch.dict(os.environ, {"VERSION_FILE": "custom_version.txt"}):
            config = Config()
            assert config.VERSION_FILE == "custom_version.txt"

    def test_version_file_property_without_env_var(self):
        """Test that VERSION_FILE property returns default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.VERSION_FILE == "version.txt"

    def test_requirements_hash_file_property_with_env_var(self):
        """Test that REQUIREMENTS_HASH_FILE property returns environment variable value when set."""
        with patch.dict(os.environ, {"REQUIREMENTS_HASH_FILE": "custom_hash.txt"}):
            config = Config()
            assert config.REQUIREMENTS_HASH_FILE == "custom_hash.txt"

    def test_requirements_hash_file_property_without_env_var(self):
        """Test that REQUIREMENTS_HASH_FILE property returns default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.REQUIREMENTS_HASH_FILE == "requirements_hash.txt"

    def test_build_script_property_with_env_var(self):
        """Test that BUILD_SCRIPT property returns environment variable value when set."""
        with patch.dict(os.environ, {"BUILD_SCRIPT": "custom_build.sh"}):
            config = Config()
            assert config.BUILD_SCRIPT == "custom_build.sh"

    def test_build_script_property_without_env_var(self):
        """Test that BUILD_SCRIPT property returns default value when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.BUILD_SCRIPT == "build_and_publish.sh"

    def test_get_model_version_with_existing_file(self, temp_files):
        """Test that get_model_version reads version from existing file."""
        with patch.dict(os.environ, {"VERSION_FILE": str(temp_files["version_file"])}):
            config = Config()
            version = config.get_model_version()
            assert version == "1.0.0"

    def test_get_model_version_with_missing_file(self):
        """Test that get_model_version raises FileNotFoundError when version file doesn't exist."""
        with patch.dict(os.environ, {"VERSION_FILE": "nonexistent_file.txt"}):
            config = Config()
            with pytest.raises(FileNotFoundError, match="Version file nonexistent_file.txt not found"):
                config.get_model_version()

    def test_get_data_paths(self, temp_files):
        """Test that get_data_paths returns correct data paths."""
        with patch.dict(
            os.environ,
            {
                "VERSION_FILE": str(temp_files["version_file"]),
                "MODEL_NAME": "test-model",
                "S3_BUCKET_NAME": "test-bucket",
            },
        ):
            config = Config()
            data_paths = config.get_data_paths()

            expected = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            assert data_paths == expected

    def test_get_deployment_bucket(self):
        """Test that get_deployment_bucket returns correct bucket name for different environments."""
        with patch.dict(os.environ, {"DEPLOYMENT_BUCKET_PREFIX": "sagemaker-models"}):
            config = Config()
            assert config.get_deployment_bucket("dev") == "sagemaker-models-dev"
            assert config.get_deployment_bucket("staging") == "sagemaker-models-staging"
            assert config.get_deployment_bucket("production") == "sagemaker-models-production"

    def test_get_endpoint_name(self, temp_files):
        """Test that get_endpoint_name returns correct endpoint name."""
        with patch.dict(os.environ, {"VERSION_FILE": str(temp_files["version_file"]), "MODEL_NAME": "test-model"}):
            config = Config()
            assert config.get_endpoint_name("dev") == "test-model-v1-0-0-dev"
            assert config.get_endpoint_name("staging") == "test-model-v1-0-0-staging"
            assert config.get_endpoint_name("production") == "test-model-v1-0-0-production"

    def test_get_ecr_image_uri_with_custom_tag(self, mock_aws_services):
        """Test that get_ecr_image_uri returns correct URI with custom tag."""
        config = Config()
        uri = config.get_ecr_image_uri("custom-tag")
        expected = "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom-tag"
        assert uri == expected

    def test_get_ecr_image_uri_without_tag(self, mock_aws_services):
        """Test that get_ecr_image_uri returns correct URI with default tag."""
        with patch.dict(os.environ, {"MODEL_NAME": "test-model"}):
            config = Config()
            uri = config.get_ecr_image_uri()
            expected = "123456789012.dkr.ecr.us-east-1.amazonaws.com/test-model"
            assert uri == expected

    def test_get_mlflow_tracking_uri(self):
        """Test that get_mlflow_tracking_uri returns the tracking URI."""
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
            config = Config()
            assert config.get_mlflow_tracking_uri() == "http://localhost:5000"

    def test_get_sagemaker_role_arn(self):
        """Test that get_sagemaker_role_arn returns the role ARN."""
        with patch.dict(os.environ, {"SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789012:role/test-role"}):
            config = Config()
            assert config.get_sagemaker_role_arn() == "arn:aws:iam::123456789012:role/test-role"


class TestEnvironmentConfigs:
    """Test cases for environment-specific configuration classes."""

    def test_dev_config_inheritance(self):
        """Test that DevConfig inherits from Config."""
        config = DevConfig()
        assert isinstance(config, Config)

    def test_staging_config_inheritance(self):
        """Test that StagingConfig inherits from Config."""
        config = StagingConfig()
        assert isinstance(config, Config)

    def test_production_config_inheritance(self):
        """Test that ProductionConfig inherits from Config."""
        config = ProductionConfig()
        assert isinstance(config, Config)


class TestGetConfig:
    """Test cases for the get_config factory function."""

    def test_get_config_dev(self):
        """Test that get_config returns DevConfig for 'dev' environment."""
        config = get_config("dev")
        assert isinstance(config, DevConfig)

    def test_get_config_staging(self):
        """Test that get_config returns StagingConfig for 'staging' environment."""
        config = get_config("staging")
        assert isinstance(config, StagingConfig)

    def test_get_config_production(self):
        """Test that get_config returns ProductionConfig for 'production' environment."""
        config = get_config("production")
        assert isinstance(config, ProductionConfig)

    def test_get_config_prod_alias(self):
        """Test that get_config returns ProductionConfig for 'prod' alias."""
        config = get_config("prod")
        assert isinstance(config, ProductionConfig)

    def test_get_config_unknown_environment(self):
        """Test that get_config returns DevConfig for unknown environment."""
        config = get_config("unknown")
        assert isinstance(config, DevConfig)

    def test_get_config_case_insensitive(self):
        """Test that get_config is case insensitive."""
        config_upper = get_config("PRODUCTION")
        config_lower = get_config("production")
        assert isinstance(config_upper, ProductionConfig)
        assert isinstance(config_lower, ProductionConfig)
