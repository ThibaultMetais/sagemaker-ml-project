"""
Pytest configuration and common fixtures for the SageMaker project tests.

This module provides shared fixtures, mocks, and test utilities that are used
across all test modules to ensure consistent testing behavior and reduce code duplication.
"""

import os

# Add src to Python path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pandas as pd
import pytest
from moto import mock_aws

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data():
    """
    Fixture providing sample test data for training and testing.

    Returns:
        dict: Dictionary containing sample training and test DataFrames
    """
    # Create sample data similar to the dry bean dataset
    sample_data = {
        "Area": [100, 200, 150, 300, 250, 180, 220, 280, 120, 160],
        "MajorAxisLength": [10, 15, 12, 18, 16, 13, 14, 17, 11, 12],
        "MinorAxisLength": [8, 12, 10, 14, 13, 9, 11, 15, 7, 9],
        "Eccentricity": [0.6, 0.7, 0.65, 0.8, 0.75, 0.68, 0.72, 0.78, 0.62, 0.66],
        "Roundness": [0.8, 0.7, 0.75, 0.6, 0.65, 0.72, 0.68, 0.58, 0.82, 0.74],
        "Class": ["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"],
    }

    df = pd.DataFrame(sample_data)

    # Split into train and test (70/30 split)
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    return {"train": train_df, "test": test_df, "full": df}


@pytest.fixture
def mock_env_vars():
    """
    Fixture that mocks environment variables required by the Config class.

    This fixture sets up all necessary environment variables for testing
    without requiring actual AWS credentials or external services.
    """
    env_vars = {
        "AWS_REGION": "us-east-1",
        "S3_BUCKET_NAME": "test-bucket",
        "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789012:role/test-role",
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MODEL_NAME": "test-model",
        "DEFAULT_TRAINING_INSTANCE_TYPE": "ml.c5.xlarge",
        "DEFAULT_INFERENCE_INSTANCE_TYPE": "ml.m5.large",
        "DEFAULT_INSTANCE_COUNT": "1",
        "N_ESTIMATORS": "100",
        "MIN_SAMPLES_LEAF": "3",
        "VERSION_FILE": "version.txt",
        "REQUIREMENTS_HASH_FILE": "requirements_hash.txt",
        "BUILD_SCRIPT": "build_and_publish.sh",
        "DEPLOYMENT_BUCKET_PREFIX": "sagemaker-models",
        "ECR_TRAINING_REPO_NAME": "sagemaker/training_images",
    }

    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_aws_services():
    """
    Fixture that mocks AWS services using moto.

    This fixture provides mocked versions of AWS services including STS, S3, and ECR
    to allow testing of AWS-dependent functionality without requiring real AWS credentials.
    """
    with mock_aws():
        # Mock STS to return a fake account ID
        sts = boto3.client("sts")
        sts.get_caller_identity = MagicMock(return_value={"Account": "123456789012"})

        # Create test S3 bucket
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="test-bucket")

        # Create test ECR repository
        ecr = boto3.client("ecr")
        ecr.create_repository(repositoryName="test-model")

        yield {"sts": sts, "s3": s3, "ecr": ecr}


@pytest.fixture
def mock_mlflow():
    """
    Fixture that mocks MLflow tracking and model registry functionality.

    This fixture provides mocked versions of MLflow components to allow testing
    of MLflow-dependent functionality without requiring a real MLflow server.
    """
    with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"), patch("mlflow.start_run"), patch(
        "mlflow.log_params"
    ), patch("mlflow.log_metrics"), patch("mlflow.sklearn.log_model"), patch("mlflow.register_model"), patch(
        "mlflow.tracking.MlflowClient"
    ), patch("mlflow.sklearn.load_model"), patch("mlflow.models.infer_signature"), patch(
        "mlflow.deployments.get_deploy_client"
    ):
        # Mock MLflow client
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.tags = {"semver": "1.0.0"}

        mock_client.search_model_versions.return_value = [mock_version]
        mock_client.get_model_version.return_value = mock_version
        mock_client.set_model_version_tag.return_value = None

        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            yield mock_client


@pytest.fixture
def temp_files():
    """
    Fixture that creates temporary files for testing file operations.

    This fixture creates temporary files that can be used for testing
    file reading, writing, and manipulation operations.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create version file
        version_file = temp_path / "version.txt"
        version_file.write_text("1.0.0")

        # Create requirements hash file
        hash_file = temp_path / "requirements_hash.txt"
        hash_file.write_text("test_hash")

        # Create pyproject.toml
        pyproject_file = temp_path / "pyproject.toml"
        pyproject_file.write_text("[project]\nname = 'test-project'")

        # Create uv.lock
        uv_lock_file = temp_path / "uv.lock"
        uv_lock_file.write_text("test-lock-content")

        # Create requirements.txt
        requirements_file = temp_path / "requirements.txt"
        requirements_file.write_text("pandas>=1.5.0\nnumpy>=1.21.0")

        yield {
            "temp_dir": temp_path,
            "version_file": version_file,
            "hash_file": hash_file,
            "pyproject_file": pyproject_file,
            "uv_lock_file": uv_lock_file,
            "requirements_file": requirements_file,
            "build_script": temp_path / "build_and_publish.sh",
        }


@pytest.fixture
def mock_subprocess():
    """
    Fixture that mocks subprocess calls to prevent actual command execution during tests.

    This fixture mocks subprocess.run to return successful results without
    actually executing any commands, which is useful for testing functions
    that call external processes.
    """
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.check_returncode = MagicMock()
        yield mock_run


@pytest.fixture
def mock_docker():
    """
    Fixture that mocks Docker operations to prevent actual Docker commands during tests.

    This fixture mocks Docker client operations to allow testing of Docker-related
    functionality without requiring Docker to be installed or running.
    """
    with patch("docker.from_env") as mock_docker_client:
        mock_image = MagicMock()
        mock_docker_client.return_value.images.get.return_value = mock_image
        yield mock_docker_client


@pytest.fixture
def mock_sagemaker_session():
    """
    Fixture that mocks SageMaker session for testing SageMaker-dependent functionality.

    This fixture provides a mocked SageMaker session that can be used to test
    functions that interact with SageMaker without requiring actual AWS resources.
    """
    with patch("sagemaker.Session") as mock_session_class:
        mock_session = MagicMock()
        mock_session.upload_data.return_value = "s3://test-bucket/test-key"
        mock_session_class.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_estimator():
    """
    Fixture that mocks SageMaker Estimator for testing estimator-related functionality.

    This fixture provides a mocked SageMaker Estimator that can be used to test
    functions that create or interact with SageMaker estimators.
    """
    with patch("sagemaker.estimator.Estimator") as mock_estimator_class:
        mock_estimator = MagicMock()
        mock_estimator.fit.return_value = None
        mock_estimator_class.return_value = mock_estimator
        yield mock_estimator


@pytest.fixture
def mock_hyperparameter_tuner():
    """
    Fixture that mocks SageMaker HyperparameterTuner for testing tuning functionality.

    This fixture provides a mocked SageMaker HyperparameterTuner that can be used
    to test functions that create or interact with hyperparameter tuning jobs.
    """
    with patch("sagemaker.tuner.HyperparameterTuner") as mock_tuner_class:
        mock_tuner = MagicMock()
        mock_tuner.fit.return_value = None
        mock_tuner.analytics.return_value.dataframe.return_value = pd.DataFrame()
        mock_tuner.best_estimator.return_value = MagicMock()
        mock_tuner_class.return_value = mock_tuner
        yield mock_tuner


@pytest.fixture
def mock_ucimlrepo():
    """Mock UCI ML repository to avoid actual data fetching."""
    with patch("prepare_data.fetch_ucirepo") as mock_fetch:
        real_df = pd.DataFrame(
            {
                "Area": [100, 200, 300, 400, 500],
                "MajorAxisLength": [10, 20, 30, 40, 50],
                "MinorAxisLength": [5, 10, 15, 20, 25],
                "Eccentricity": [0.1, 0.2, 0.3, 0.4, 0.5],
                "Roundness": [0.8, 0.7, 0.6, 0.5, 0.4],
                "Class": ["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER"],
            }
        )
        # Return a real dict, not a MagicMock
        mock_fetch.return_value = {"data": {"original": real_df}}
        yield mock_fetch


@pytest.fixture
def mock_matplotlib():
    """
    Fixture that mocks matplotlib to prevent actual plot generation during tests.

    This fixture mocks matplotlib.pyplot to prevent actual plot generation
    which can cause issues in headless environments or CI/CD pipelines.
    """
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.figure"), patch("seaborn.pairplot"), patch(
        "seaborn.heatmap"
    ):
        yield


@pytest.fixture
def mock_boto3_client():
    """
    Fixture that mocks boto3 client creation to prevent actual AWS API calls.

    This fixture mocks boto3.client to return mocked clients that can be used
    for testing without requiring actual AWS credentials or API access.
    """
    with patch("boto3.client") as mock_client:
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        mock_ecr = MagicMock()
        mock_ecr.describe_repositories.return_value = {"repositories": []}
        mock_ecr.describe_images.return_value = {"imageDetails": []}

        mock_client.side_effect = lambda service: {"sts": mock_sts, "ecr": mock_ecr}.get(service, MagicMock())

        yield mock_client
