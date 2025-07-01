"""
Unit tests for the deploy.py module.

This module contains comprehensive tests for model deployment functionality,
including ECR operations, Docker container building, and SageMaker deployment.
"""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from deploy import (
    build_mlflow_container,
    deploy_to_sagemaker,
    ensure_ecr_repo_exists,
    get_model_uri_from_semver,
    image_exists_in_ecr,
)


class TestEcrOperations:
    """Test cases for ECR-related operations."""

    def test_ensure_ecr_repo_exists_when_exists(self):
        """Test ensure_ecr_repo_exists when repository already exists."""
        ecr_client = MagicMock()
        ecr_client.describe_repositories.return_value = {"repositories": [{"repositoryName": "test-repo"}]}

        ensure_ecr_repo_exists(ecr_client, "test-repo")

        ecr_client.describe_repositories.assert_called_once_with(repositoryNames=["test-repo"])
        ecr_client.create_repository.assert_not_called()

    def test_ensure_ecr_repo_exists_when_missing(self):
        """Test ensure_ecr_repo_exists when repository doesn't exist."""
        ecr_client = MagicMock()
        ecr_client.describe_repositories.side_effect = ClientError(
            {"Error": {"Code": "RepositoryNotFoundException"}}, "DescribeRepositories"
        )

        ensure_ecr_repo_exists(ecr_client, "test-repo")

        ecr_client.describe_repositories.assert_called_once_with(repositoryNames=["test-repo"])
        ecr_client.create_repository.assert_called_once_with(repositoryName="test-repo")

    def test_image_exists_in_ecr_when_exists(self):
        """Test image_exists_in_ecr when image exists."""
        ecr_client = MagicMock()
        ecr_client.describe_images.return_value = {"imageDetails": [{"imageTag": "test-tag"}]}

        result = image_exists_in_ecr(ecr_client, "test-repo", "test-tag")
        assert result is True

    def test_image_exists_in_ecr_when_missing(self):
        """Test image_exists_in_ecr when image doesn't exist."""
        ecr_client = MagicMock()
        ecr_client.describe_images.side_effect = ClientError(
            {"Error": {"Code": "ImageNotFoundException"}}, "DescribeImages"
        )

        result = image_exists_in_ecr(ecr_client, "test-repo", "test-tag")
        assert result is False


class TestMlflowOperations:
    """Test cases for MLflow-related operations."""

    def test_get_model_uri_from_semver_success(self):
        """Test get_model_uri_from_semver when model version is found."""
        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.version = 1
        mock_version.tags = {"semver": "1.0.0"}

        mock_client.search_model_versions.return_value = [mock_version]

        with patch("deploy.MlflowClient", return_value=mock_client):
            result = get_model_uri_from_semver("test-model", "1.0.0")
            assert result == "models:/test-model/1"

    def test_get_model_uri_from_semver_not_found(self):
        """Test get_model_uri_from_semver when model version is not found."""
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = []

        with patch("deploy.MlflowClient", return_value=mock_client), pytest.raises(
            Exception,
            match="No model version found with semver tag: 1.0.0",
        ):
            get_model_uri_from_semver("test-model", "1.0.0")


class TestDockerOperations:
    """Test cases for Docker-related operations."""

    def test_build_mlflow_container(self, temp_files, mock_subprocess):
        """Test build_mlflow_container function."""
        mock_subprocess.return_value.returncode = 0

        with patch("deploy.Path") as mock_path:
            mock_path.return_value.parent.parent = temp_files["temp_dir"]

            build_mlflow_container("models:/test-model/1", "test-image")

            assert mock_subprocess.call_count == 2


class TestSageMakerDeployment:
    """Test cases for SageMaker deployment operations."""

    def test_deploy_to_sagemaker(self):
        """Test deploy_to_sagemaker function."""
        mock_client = MagicMock()

        with patch("deploy.mlflow.deployments.get_deploy_client", return_value=mock_client):
            deploy_to_sagemaker(
                model_uri="models:/test-model/1",
                endpoint_name="test-endpoint",
                image_url="test-image-url",
                role_arn="arn:aws:iam::123456789012:role/test-role",
                bucket="test-bucket",
                region="us-east-1",
                instance_type="ml.m5.large",
                instance_count=2,
            )

            mock_client.create_deployment.assert_called_once()

    def test_deploy_to_sagemaker_default_instance_count(self, mock_mlflow):
        """Test deploy_to_sagemaker with default instance count."""
        mock_client = MagicMock()

        with patch("deploy.mlflow.deployments.get_deploy_client", return_value=mock_client):
            deploy_to_sagemaker(
                model_uri="models:/test-model/1",
                endpoint_name="test-endpoint",
                image_url="test-image-url",
                role_arn="arn:aws:iam::123456789012:role/test-role",
                bucket="test-bucket",
                region="us-east-1",
                instance_type="ml.m5.large",
            )

            call_args = mock_client.create_deployment.call_args
            assert call_args[1]["config"]["instance_count"] == 1


class TestMainDeploymentFlow:
    """Test cases for the main deployment flow."""

    def test_main_deployment_flow_success(
        self, mock_env_vars, mock_aws_services, mock_mlflow, mock_subprocess, mock_docker, temp_files
    ):
        """Test the complete deployment flow."""
        # Patch Config class instead of deploy.config
        with patch("config.Config") as mock_config:
            mock_config_instance = mock_config.return_value
            mock_config_instance.MODEL_NAME = "test-model"
            mock_config_instance.DEFAULT_INFERENCE_INSTANCE_TYPE = "ml.m5.large"
            mock_config_instance.DEFAULT_INSTANCE_COUNT = 1
            mock_config_instance.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config_instance.get_model_version.return_value = "1.0.0"
            mock_config_instance.AWS_REGION = "us-east-1"
            mock_config_instance.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
            mock_config_instance.get_endpoint_name.return_value = "test-endpoint"
            mock_config_instance.get_deployment_bucket.return_value = "test-bucket"
            mock_config_instance.AWS_ACCOUNT_ID = "123456789012"
            mock_config_instance.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"

            # Mock argument parser and all functions
            with patch("deploy.argparse.ArgumentParser") as mock_parser_class, patch(
                "deploy.get_model_uri_from_semver"
            ) as mock_get_uri, patch("deploy.docker_login_to_ecr") as mock_login, patch(
                "deploy.ensure_ecr_repo_exists"
            ) as mock_ensure_repo, patch("deploy.image_exists_in_ecr") as mock_image_exists, patch(
                "deploy.build_mlflow_container"
            ) as mock_build, patch("deploy.tag_and_push_image") as mock_tag_push, patch(
                "deploy.deploy_to_sagemaker"
            ) as mock_deploy, patch("deploy.boto3.client") as mock_boto3, patch("deploy.mlflow.set_tracking_uri"):
                mock_parser = MagicMock()
                mock_args = MagicMock()
                mock_args.env = "dev"
                mock_args.instance_type = "ml.m5.large"
                mock_args.instance_count = 1
                mock_parser.parse_args.return_value = mock_args
                mock_parser_class.return_value = mock_parser

                mock_get_uri.return_value = "models:/test-model/1"
                mock_image_exists.return_value = False

                mock_ecr = MagicMock()
                mock_boto3.return_value = mock_ecr

                # Import and run the main block
                import deploy

                deploy.main()
                # Verify the deployment flow
                mock_get_uri.assert_called_once_with("test-model", "2.0.0")
                mock_login.assert_called_once_with(mock_ecr.get_caller_identity()["Account"], "us-east-1")
                mock_ensure_repo.assert_called_once_with(mock_ecr, "test-model")
                mock_image_exists.assert_called_once_with(mock_ecr, "test-model", "2.0.0")
                mock_build.assert_called_once()
                mock_tag_push.assert_called_once()
                mock_deploy.assert_called_once()

    def test_main_deployment_flow_image_exists(
        self, mock_env_vars, mock_aws_services, mock_mlflow, mock_subprocess, mock_docker, temp_files
    ):
        """Test deployment flow when image already exists."""
        # Patch Config class instead of deploy.config
        with patch("config.Config") as mock_config:
            mock_config_instance = mock_config.return_value
            mock_config_instance.MODEL_NAME = "test-model"
            mock_config_instance.DEFAULT_INFERENCE_INSTANCE_TYPE = "ml.m5.large"
            mock_config_instance.DEFAULT_INSTANCE_COUNT = 1
            mock_config_instance.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config_instance.get_model_version.return_value = "1.0.0"
            mock_config_instance.AWS_REGION = "us-east-1"
            mock_config_instance.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
            mock_config_instance.get_endpoint_name.return_value = "test-endpoint"
            mock_config_instance.get_deployment_bucket.return_value = "test-bucket"
            mock_config_instance.AWS_ACCOUNT_ID = "123456789012"
            mock_config_instance.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"

            # Mock argument parser and all functions
            with patch("deploy.argparse.ArgumentParser") as mock_parser_class, patch(
                "deploy.get_model_uri_from_semver"
            ) as mock_get_uri, patch("deploy.docker_login_to_ecr") as mock_login, patch(
                "deploy.ensure_ecr_repo_exists"
            ) as mock_ensure_repo, patch("deploy.image_exists_in_ecr") as mock_image_exists, patch(
                "deploy.build_mlflow_container"
            ) as mock_build, patch("deploy.tag_and_push_image") as mock_tag_push, patch(
                "deploy.deploy_to_sagemaker"
            ) as mock_deploy, patch("deploy.boto3.client") as mock_boto3, patch("deploy.mlflow.set_tracking_uri"):
                mock_parser = MagicMock()
                mock_args = MagicMock()
                mock_args.env = "dev"
                mock_args.instance_type = "ml.m5.large"
                mock_args.instance_count = 1
                mock_parser.parse_args.return_value = mock_args
                mock_parser_class.return_value = mock_parser

                mock_get_uri.return_value = "models:/test-model/1"
                mock_image_exists.return_value = True  # Image already exists

                mock_ecr = MagicMock()
                mock_boto3.return_value = mock_ecr

                # Import and run the main block
                import deploy

                deploy.main()
                # Verify the deployment flow
                mock_get_uri.assert_called_once_with("test-model", "2.0.0")
                mock_login.assert_called_once_with(mock_ecr.get_caller_identity()["Account"], "us-east-1")
                mock_ensure_repo.assert_called_once_with(mock_ecr, "test-model")
                mock_image_exists.assert_called_once_with(mock_ecr, "test-model", "2.0.0")
                # Should skip building and pushing
                mock_build.assert_not_called()
                mock_tag_push.assert_not_called()
                # Should still deploy
                mock_deploy.assert_called_once()
