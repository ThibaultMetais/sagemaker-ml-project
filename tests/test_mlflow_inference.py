"""
Unit tests for the test_mlflow_inference.py module.

This module contains comprehensive tests for MLflow model inference functionality,
including model loading from runs and model registry, and prediction execution.
"""

from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest

from test_mlflow_inference import test_mlflow_inference


class TestMlflowInference:
    """Test cases for the test_mlflow_inference function."""

    def test_test_mlflow_inference_with_run_id(self, mock_mlflow, test_data):
        """Test inference with a specific run ID."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}

            # Mock MLflow client
            mock_client = MagicMock()

            # Mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = [0, 1, 0]

            with patch("test_mlflow_inference.mlflow.set_tracking_uri") as mock_set_uri, patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), patch("test_mlflow_inference.mlflow.sklearn.load_model", return_value=mock_model), patch(
                "test_mlflow_inference.pd.read_csv"
            ) as mock_read_csv:
                # Mock test data
                mock_read_csv.return_value = test_data["test"]

                # Call function with run ID
                test_mlflow_inference(run_id="test-run-id")

                # Verify MLflow was configured
                mock_set_uri.assert_called_once_with("http://localhost:5000")

                # Verify model was loaded from run
                mlflow.sklearn.load_model.assert_called_once_with("runs:/test-run-id/model")

                # Verify predictions were made
                mock_model.predict.assert_called_once()

    def test_test_mlflow_inference_without_run_id(self, mock_mlflow, test_data):
        """Test inference without run ID (using model registry)."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}
            mock_config.get_model_version.return_value = "1.0.0"

            # Mock MLflow client
            mock_client = MagicMock()
            mock_version = MagicMock()
            mock_version.version = 1
            mock_version.tags = {"semver": "1.0.0"}

            mock_client.search_model_versions.return_value = [mock_version]

            # Mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = [0, 1, 0]

            with patch("test_mlflow_inference.mlflow.set_tracking_uri") as mock_set_uri, patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), patch("test_mlflow_inference.mlflow.sklearn.load_model", return_value=mock_model), patch(
                "test_mlflow_inference.pd.read_csv"
            ) as mock_read_csv:
                # Mock test data
                mock_read_csv.return_value = test_data["test"]

                # Call function without run ID
                test_mlflow_inference()

                # Verify config was used correctly
                mock_config.get_mlflow_tracking_uri.assert_called_once()
                mock_config.get_model_version.assert_called_once()

                # Verify MLflow was configured with correct URI
                mock_set_uri.assert_called_once_with("http://localhost:5000")

                # Verify model was loaded from registry
                mlflow.sklearn.load_model.assert_called_once_with("models:/test-model/1")

                # Verify predictions were made
                mock_model.predict.assert_called_once()

    def test_test_mlflow_inference_model_not_found(self, mock_mlflow):
        """Test inference when model is not found in registry."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}
            mock_config.get_model_version.return_value = "1.0.0"

            # Mock MLflow client
            mock_client = MagicMock()
            mock_client.search_model_versions.return_value = []

            with patch("test_mlflow_inference.mlflow.set_tracking_uri"), patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), pytest.raises(ValueError, match="No version of model 'test-model' with tag 'semver=1.0.0' found."):
                # Call function without run ID
                test_mlflow_inference()

    def test_test_mlflow_inference_wrong_semver(self, mock_mlflow):
        """Test inference when semver tag doesn't match."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}
            mock_config.get_model_version.return_value = "1.0.0"

            # Mock MLflow client
            mock_client = MagicMock()
            mock_version = MagicMock()
            mock_version.version = 1
            mock_version.tags = {"semver": "2.0.0"}  # Different semver

            mock_client.search_model_versions.return_value = [mock_version]

            with patch("test_mlflow_inference.mlflow.set_tracking_uri"), patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), pytest.raises(ValueError, match="No version of model 'test-model' with tag 'semver=1.0.0' found."):
                # Call function without run ID
                test_mlflow_inference()

    def test_test_mlflow_inference_data_loading(self, mock_mlflow, test_data):
        """Test that test data is loaded correctly."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}

            # Mock MLflow client
            mock_client = MagicMock()

            # Mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = [0, 1, 0]

            with patch("test_mlflow_inference.mlflow.set_tracking_uri"), patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), patch("test_mlflow_inference.mlflow.sklearn.load_model", return_value=mock_model), patch(
                "test_mlflow_inference.pd.read_csv"
            ) as mock_read_csv:
                # Mock test data
                mock_read_csv.return_value = test_data["test"]

                # Call function with run ID
                test_mlflow_inference(run_id="test-run-id")

                # Verify data was loaded from correct path
                mock_read_csv.assert_called_once_with(
                    "s3://oppizi-ml/my-example-project/2.0.0/data/test/dry-bean-test.csv"
                )

                # Verify model was called with correct data (without Class column)
                expected_data = test_data["test"].drop("Class", axis=1)
                # Use pandas testing utilities for DataFrame comparison
                pd.testing.assert_frame_equal(mock_model.predict.call_args[0][0], expected_data, check_dtype=False)

    def test_test_mlflow_inference_prediction_output(self, mock_mlflow, test_data):
        """Test that predictions are made and output correctly."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}

            # Mock MLflow client
            mock_client = MagicMock()

            # Mock model with specific predictions
            mock_model = MagicMock()
            expected_predictions = [0, 1, 0, 1, 0]
            mock_model.predict.return_value = expected_predictions

            with patch("test_mlflow_inference.mlflow.set_tracking_uri"), patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), patch("test_mlflow_inference.mlflow.sklearn.load_model", return_value=mock_model), patch(
                "test_mlflow_inference.pd.read_csv"
            ) as mock_read_csv, patch("test_mlflow_inference.print") as mock_print:
                # Mock test data
                mock_read_csv.return_value = test_data["test"]

                # Call function with run ID
                test_mlflow_inference(run_id="test-run-id")

                # Verify predictions were made
                mock_model.predict.assert_called_once()

                # Verify output was printed
                mock_print.assert_called_with("Predictions: ", expected_predictions)

    def test_test_mlflow_inference_error_handling_model_loading(self, mock_mlflow):
        """Test error handling when model loading fails."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}

            # Mock MLflow client
            mock_client = MagicMock()

            with patch("test_mlflow_inference.mlflow.set_tracking_uri"), patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), patch("test_mlflow_inference.mlflow.sklearn.load_model") as mock_load_model:
                # Mock model loading to fail
                mock_load_model.side_effect = Exception("Model loading failed")

                # Call function with run ID
                with pytest.raises(Exception, match="Model loading failed"):
                    test_mlflow_inference(run_id="test-run-id")

    def test_test_mlflow_inference_error_handling_data_loading(self, mock_mlflow):
        """Test error handling when data loading fails."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}

            # Mock MLflow client
            mock_client = MagicMock()

            # Mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = [0, 1, 0]

            with patch("test_mlflow_inference.mlflow.set_tracking_uri"), patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), patch("test_mlflow_inference.mlflow.sklearn.load_model", return_value=mock_model), patch(
                "test_mlflow_inference.pd.read_csv"
            ) as mock_read_csv:
                # Mock data loading to fail
                mock_read_csv.side_effect = Exception("Data loading failed")

                # Call function with run ID
                with pytest.raises(Exception, match="Data loading failed"):
                    test_mlflow_inference(run_id="test-run-id")

    def test_test_mlflow_inference_error_handling_prediction(self, mock_mlflow, test_data):
        """Test error handling when prediction fails."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}

            # Mock MLflow client
            mock_client = MagicMock()

            # Mock model that raises an exception
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception("Prediction failed")

            with patch("test_mlflow_inference.mlflow.set_tracking_uri"), patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), patch("test_mlflow_inference.mlflow.sklearn.load_model", return_value=mock_model), patch(
                "test_mlflow_inference.pd.read_csv"
            ) as mock_read_csv:
                # Mock test data
                mock_read_csv.return_value = test_data["test"]

                # Call function with run ID and expect exception
                with pytest.raises(Exception, match="Prediction failed"):
                    test_mlflow_inference(run_id="test-run-id")

    def test_test_mlflow_inference_config_integration(self, mock_mlflow, test_data):
        """Test that config integration works correctly."""
        # Mock config
        with patch("test_mlflow_inference.config") as mock_config:
            mock_config.get_mlflow_tracking_uri.return_value = "http://localhost:5000"
            mock_config.MODEL_NAME = "test-model"
            mock_config.get_data_paths.return_value = {"test_data_path": "s3://test-bucket/test-model/1.0.0/data/test"}
            mock_config.get_model_version.return_value = "1.0.0"

            # Mock MLflow client
            mock_client = MagicMock()
            mock_version = MagicMock()
            mock_version.version = 1
            mock_version.tags = {"semver": "1.0.0"}

            mock_client.search_model_versions.return_value = [mock_version]

            # Mock model
            mock_model = MagicMock()
            mock_model.predict.return_value = [0, 1, 0]

            with patch("test_mlflow_inference.mlflow.set_tracking_uri") as mock_set_uri, patch(
                "test_mlflow_inference.mlflow.tracking.MlflowClient", return_value=mock_client
            ), patch("test_mlflow_inference.mlflow.sklearn.load_model", return_value=mock_model), patch(
                "test_mlflow_inference.pd.read_csv"
            ) as mock_read_csv:
                # Mock test data
                mock_read_csv.return_value = test_data["test"]

                # Call function without run ID
                test_mlflow_inference()

                # Verify config was used correctly
                mock_config.get_mlflow_tracking_uri.assert_called_once()
                mock_config.get_model_version.assert_called_once()

                # Verify MLflow was configured with correct URI
                mock_set_uri.assert_called_once_with("http://localhost:5000")

                # Verify model was loaded from registry
                mlflow.sklearn.load_model.assert_called_once_with("models:/test-model/1")
