"""
Unit tests for the estimator.py module.

This module contains comprehensive tests for SageMaker estimator creation,
hyperparameter configuration, and build script integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from estimator import build_estimator


class TestBuildEstimator:
    """Test cases for the build_estimator function."""

    def test_build_estimator_basic(self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess):
        """Test basic estimator creation without model registration."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Call build_estimator
                        result = build_estimator(register_model=False)

                        # Verify result
                        assert result == mock_estimator

                        # Verify Estimator was created with correct parameters
                        mock_estimator_class.assert_called_once()
                        call_args = mock_estimator_class.call_args

                        # Check image URI
                        expected_image_uri = (
                            "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker/training_images:test-model"
                        )
                        assert call_args[1]["image_uri"] == expected_image_uri

                        # Check role
                        assert call_args[1]["role"] == "arn:aws:iam::123456789012:role/test-role"

                        # Check source directory and entry point
                        assert call_args[1]["source_dir"] == "src"
                        assert call_args[1]["entry_point"] == "training.py"

                        # Check instance configuration
                        assert call_args[1]["instance_count"] == 1
                        assert call_args[1]["instance_type"] == "ml.c5.xlarge"

                        # Check base job name
                        assert call_args[1]["base_job_name"] == "test-model-training"

                        # Check hyperparameters
                        expected_hyperparams = {"n-estimators": 100, "min-samples-leaf": 3, "register-model": False}
                        assert call_args[1]["hyperparameters"] == expected_hyperparams

                        # Check environment variables
                        expected_env = {
                            "AWS_REGION": "us-east-1",
                            "MLFLOW_TRACKING_URI": "http://localhost:5000",
                            "MODEL_NAME": "test-model",
                            "S3_BUCKET_NAME": "test-bucket",
                            "N_ESTIMATORS": "100",
                            "MIN_SAMPLES_LEAF": "3",
                            "VERSION_FILE": "version.txt",
                            "REQUIREMENTS_HASH_FILE": "requirements_hash.txt",
                        }
                        assert call_args[1]["environment"] == expected_env

                        # Check dependencies
                        assert len(call_args[1]["dependencies"]) == 2
                        dependencies = call_args[1]["dependencies"]
                        # Convert all dependencies to strings for comparison
                        str_dependencies = [str(dep) for dep in dependencies]
                        assert str(temp_files["requirements_file"]) in str_dependencies
                        assert str(temp_files["version_file"]) in str_dependencies

    def test_build_estimator_with_model_registration(
        self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess
    ):
        """Test estimator creation with model registration enabled."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Call build_estimator with registration
                        result = build_estimator(register_model=True)

                        # Verify result
                        assert result == mock_estimator

                        # Verify Estimator was created with registration hyperparameter
                        mock_estimator_class.assert_called_once()
                        call_args = mock_estimator_class.call_args

                        # Check hyperparameters include registration flag
                        expected_hyperparams = {"n-estimators": 100, "min-samples-leaf": 3, "register-model": True}
                        assert call_args[1]["hyperparameters"] == expected_hyperparams

    def test_build_estimator_requirements_generation(
        self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess
    ):
        """Test that requirements are generated correctly."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Call build_estimator
                        build_estimator(register_model=False)

                        # Verify requirements generation was called
                        mock_gen_req.assert_called_once()
                        call_args = mock_gen_req.call_args

                        # Check that correct paths were passed
                        assert call_args[0][0] == temp_files["pyproject_file"]
                        assert call_args[0][1] == temp_files["uv_lock_file"]

    def test_build_estimator_rebuild_check(self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess):
        """Test that rebuild check is performed correctly."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Call build_estimator
                        build_estimator(register_model=False)

                        # Verify rebuild check was called
                        mock_rebuild.assert_called_once()
                        call_args = mock_rebuild.call_args

                        # Check that correct paths were passed
                        assert call_args[0][0] == str(temp_files["requirements_file"])
                        assert str(call_args[0][1]) == str(temp_files["hash_file"])
                        assert str(call_args[0][2]) == str(temp_files["build_script"])

    def test_build_estimator_hyperparameter_copy(self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess):
        """Test that hyperparameters are properly copied and modified."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 200, "min-samples-leaf": 5}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Call build_estimator
                        build_estimator(register_model=True)

                        # Verify Estimator was created with correct hyperparameters
                        mock_estimator_class.assert_called_once()
                        call_args = mock_estimator_class.call_args

                        # Check that original hyperparameters were copied and modified
                        expected_hyperparams = {"n-estimators": 200, "min-samples-leaf": 5, "register-model": True}
                        assert call_args[1]["hyperparameters"] == expected_hyperparams

    def test_build_estimator_environment_variables(self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess):
        """Test that environment variables are set correctly."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-west-2"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://mlflow:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Call build_estimator
                        build_estimator(register_model=False)

                        # Verify Estimator was created with correct environment variables
                        mock_estimator_class.assert_called_once()
                        call_args = mock_estimator_class.call_args

                        expected_env = {
                            "AWS_REGION": "us-west-2",
                            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
                            "MODEL_NAME": "test-model",
                            "S3_BUCKET_NAME": "test-bucket",
                            "N_ESTIMATORS": "100",
                            "MIN_SAMPLES_LEAF": "3",
                            "VERSION_FILE": "version.txt",
                            "REQUIREMENTS_HASH_FILE": "requirements_hash.txt",
                        }
                        assert call_args[1]["environment"] == expected_env

    def test_build_estimator_dependencies(self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess):
        """Test that dependencies are set correctly."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Call build_estimator
                        build_estimator(register_model=False)

                        # Verify Estimator was created with correct dependencies
                        mock_estimator_class.assert_called_once()
                        call_args = mock_estimator_class.call_args

                        # Verify dependencies
                        dependencies = call_args[1]["dependencies"]
                        # Convert all dependencies to strings for comparison
                        str_dependencies = [str(dep) for dep in dependencies]
                        assert str(temp_files["requirements_file"]) in str_dependencies
                        assert str(temp_files["version_file"]) in str_dependencies

    def test_build_estimator_error_handling_requirements_generation(
        self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess
    ):
        """Test error handling when requirements generation fails."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                # Mock requirements generation to fail
                mock_gen_req.side_effect = Exception("Requirements generation failed")
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Should raise exception
                        with pytest.raises(Exception, match="Requirements generation failed"):
                            build_estimator(register_model=False)

    def test_build_estimator_error_handling_rebuild_failure(
        self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess
    ):
        """Test error handling when rebuild check fails."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.c5.xlarge"
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                # Mock rebuild to fail
                mock_rebuild.side_effect = Exception("Rebuild failed")

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Should raise exception
                        with pytest.raises(Exception, match="Rebuild failed"):
                            build_estimator(register_model=False)

    def test_build_estimator_custom_instance_type(self, mock_env_vars, mock_aws_services, temp_files, mock_subprocess):
        """Test estimator creation with custom instance type."""
        # Mock config
        with patch("estimator.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.DEFAULT_HYPERPARAMETERS = {"n-estimators": 100, "min-samples-leaf": 3}
            mock_config.ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
            mock_config.ECR_TRAINING_REPO_NAME = "sagemaker/training_images"
            mock_config.MODEL_NAME = "test-model"
            mock_config.DEFAULT_TRAINING_INSTANCE_TYPE = "ml.p3.2xlarge"  # Custom instance type
            mock_config.AWS_REGION = "us-east-1"
            mock_config.get_sagemaker_role_arn.return_value = "arn:aws:iam::123456789012:role/test-role"
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.S3_BUCKET_NAME = "test-bucket"
            mock_config.VERSION_FILE = "version.txt"
            mock_config.REQUIREMENTS_HASH_FILE = "requirements_hash.txt"
            mock_config.BUILD_SCRIPT = "build_and_publish.sh"

            # Mock utils functions
            with patch("estimator.generate_requirements_with_uv") as mock_gen_req, patch(
                "estimator.rebuild_training_image_if_requirements_changed"
            ) as mock_rebuild:
                mock_gen_req.return_value = str(temp_files["requirements_file"])
                mock_rebuild.return_value = True

                # Mock Path operations
                with patch("estimator.Path") as mock_path:
                    mock_path.return_value.parent.parent = temp_files["temp_dir"]

                    # Mock Estimator class
                    with patch("estimator.Estimator") as mock_estimator_class:
                        mock_estimator = MagicMock()
                        mock_estimator_class.return_value = mock_estimator

                        # Call build_estimator
                        build_estimator(register_model=False)

                        # Verify Estimator was created with custom instance type
                        mock_estimator_class.assert_called_once()
                        call_args = mock_estimator_class.call_args
                        assert call_args[1]["instance_type"] == "ml.p3.2xlarge"
                        model_dir = call_args[1]["source_dir"]
                        expected_model_dir = "src"
                        assert str(model_dir) == str(expected_model_dir)
