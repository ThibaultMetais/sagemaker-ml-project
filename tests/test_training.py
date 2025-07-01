"""
Unit tests for the training.py module.

This module contains comprehensive tests for model training functionality,
including MLflow integration, model evaluation, and argument parsing.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from training import train


class TestTrainFunction:
    """Test cases for the train function."""

    def test_train_basic_functionality(self, test_data, mock_mlflow, temp_files):
        """Test basic training functionality without model registration."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            # Mock MLflow functions and all dependencies in a single with statement
            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                stack.enter_context(patch("training.infer_signature"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]

                mock_rf = MagicMock()
                mock_rf.predict.side_effect = [
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                    np.array(["SEKER", "DERMASON", "CALI"]),
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                ]
                mock_rf_class.return_value = mock_rf

                train(
                    train_path=str(train_file),
                    test_path=str(test_file),
                    params={"n_estimators": 10, "min_samples_leaf": 3},
                    model_dir=str(model_dir),
                    register_model=False,
                )

                # Verify data was read from correct files
                assert mock_read_csv.call_count == 2
                calls = mock_read_csv.call_args_list
                assert calls[0][0][0] == str(train_file)
                assert calls[1][0][0] == str(test_file)

    def test_train_with_model_registration(self, test_data, mock_mlflow, temp_files):
        """Test training with model registration enabled."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                mock_register_model = stack.enter_context(patch("training.mlflow.register_model"))
                stack.enter_context(patch("training.infer_signature"))
                mock_client_class = stack.enter_context(patch("training.mlflow.tracking.MlflowClient"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))

                # Mock MLflow client
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]

                mock_rf = MagicMock()
                mock_rf.predict.side_effect = [
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                    np.array(["SEKER", "DERMASON", "CALI"]),
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                ]
                mock_rf_class.return_value = mock_rf

                # Call train function with registration
                train(
                    train_path=str(train_file),
                    test_path=str(test_file),
                    params={"n_estimators": 10, "min_samples_leaf": 3},
                    model_dir=str(model_dir),
                    register_model=True,
                )

                # Verify model was registered
                mock_register_model.assert_called_once()
                mock_client.set_model_version_tag.assert_called_once()

    def test_train_data_loading(self, test_data, mock_mlflow, temp_files):
        """Test that data is loaded correctly from CSV files."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                stack.enter_context(patch("training.infer_signature"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]

                mock_rf = MagicMock()
                mock_rf.predict.side_effect = [
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                    np.array(["SEKER", "DERMASON", "CALI"]),
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                ]
                mock_rf_class.return_value = mock_rf

                train(
                    train_path=str(train_file),
                    test_path=str(test_file),
                    params={"n_estimators": 10, "min_samples_leaf": 3},
                    model_dir=str(model_dir),
                    register_model=False,
                )

                # Verify data was read from correct files
                assert mock_read_csv.call_count == 2
                calls = mock_read_csv.call_args_list
                assert calls[0][0][0] == str(train_file)
                assert calls[1][0][0] == str(test_file)

    def test_train_feature_extraction(self, test_data, mock_mlflow, temp_files):
        """Test that features are extracted correctly from the data."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            # Mock MLflow functions
            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                stack.enter_context(patch("training.infer_signature"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]

                mock_rf = MagicMock()
                # Mock predict to return string labels matching the test data sizes
                # predict is called 3 times: train accuracy, test accuracy, MLflow signature
                mock_rf.predict.side_effect = [
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),  # train accuracy
                    np.array(["SEKER", "DERMASON", "CALI"]),  # test accuracy
                    np.array(
                        ["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]
                    ),  # MLflow signature
                ]
                mock_rf_class.return_value = mock_rf

                train(
                    train_path=str(train_file),
                    test_path=str(test_file),
                    params={"n_estimators": 10, "min_samples_leaf": 3},
                    model_dir=str(model_dir),
                    register_model=False,
                )

                # Verify RandomForestClassifier was called with correct parameters
                mock_rf_class.assert_called_once_with(n_estimators=10, min_samples_leaf=3, n_jobs=-1)

                # Verify model was fitted
                mock_rf.fit.assert_called_once()

    def test_train_model_evaluation(self, test_data, mock_mlflow, temp_files):
        """Test that model evaluation metrics are calculated correctly."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                mock_log_metrics = stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                stack.enter_context(patch("training.infer_signature"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))
                mock_bal_acc = stack.enter_context(patch("training.balanced_accuracy_score"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]

                mock_rf = MagicMock()
                mock_rf.predict.side_effect = [
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                    np.array(["SEKER", "DERMASON", "CALI"]),
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                ]
                mock_rf_class.return_value = mock_rf

                mock_bal_acc.side_effect = [0.85, 0.80]  # train, test scores

                train(
                    train_path=str(train_file),
                    test_path=str(test_file),
                    params={"n_estimators": 10, "min_samples_leaf": 3},
                    model_dir=str(model_dir),
                    register_model=False,
                )

                # Verify metrics were logged
                mock_log_metrics.assert_called_once_with({"bal_acc_train": 0.85, "bal_acc_test": 0.80})

    def test_train_mlflow_signature_inference(self, test_data, mock_mlflow, temp_files):
        """Test that MLflow signature inference is called correctly."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                mock_infer_signature = stack.enter_context(patch("training.infer_signature"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))
                mock_bal_acc = stack.enter_context(patch("training.balanced_accuracy_score"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]

                mock_rf = MagicMock()
                mock_rf.predict.side_effect = [
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                    np.array(["SEKER", "DERMASON", "CALI"]),
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                ]
                mock_rf_class.return_value = mock_rf

                mock_bal_acc.side_effect = [0.85, 0.80]  # train, test scores

                train(
                    train_path=str(train_file),
                    test_path=str(test_file),
                    params={"n_estimators": 10, "min_samples_leaf": 3},
                    model_dir=str(model_dir),
                    register_model=False,
                )

                # Verify signature was inferred
                mock_infer_signature.assert_called_once()

    def test_train_model_saving(self, test_data, mock_mlflow, temp_files):
        """Test that the trained model is saved correctly."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                stack.enter_context(patch("training.infer_signature"))
                mock_joblib_dump = stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]

                mock_rf = MagicMock()
                mock_rf.predict.side_effect = [
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                    np.array(["SEKER", "DERMASON", "CALI"]),
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                ]
                mock_rf_class.return_value = mock_rf

                train(
                    train_path=str(train_file),
                    test_path=str(test_file),
                    params={"n_estimators": 10, "min_samples_leaf": 3},
                    model_dir=str(model_dir),
                    register_model=False,
                )

                # Verify model was saved
                mock_joblib_dump.assert_called_once_with(mock_rf, str(model_dir) + "/model.pkl")

    def test_train_error_handling_missing_files(self, mock_mlflow, temp_files):
        """Test error handling when data files are missing."""
        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                stack.enter_context(patch("training.infer_signature"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = FileNotFoundError("File not found")

                with pytest.raises(FileNotFoundError):
                    train(
                        train_path="nonexistent_train.csv",
                        test_path="nonexistent_test.csv",
                        params={"n_estimators": 10, "min_samples_leaf": 3},
                        model_dir=str(temp_files["temp_dir"] / "model"),
                        register_model=False,
                    )

    def test_train_error_handling_invalid_params(self, test_data, mock_mlflow, temp_files):
        """Test error handling with invalid model parameters."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            with ExitStack() as stack:
                stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                stack.enter_context(patch("training.infer_signature"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]
                mock_rf_class.side_effect = ValueError("Invalid parameters")

                with pytest.raises(ValueError):
                    train(
                        train_path=str(train_file),
                        test_path=str(test_file),
                        params={"n_estimators": -1, "min_samples_leaf": 0},  # Invalid params
                        model_dir=str(model_dir),
                        register_model=False,
                    )

    def test_train_mlflow_experiment_creation(self, test_data, mock_mlflow, temp_files):
        """Test that MLflow experiment is created correctly."""
        # Create test data files
        train_file = temp_files["temp_dir"] / "train.csv"
        test_file = temp_files["temp_dir"] / "test.csv"
        model_dir = temp_files["temp_dir"] / "model"

        # Save test data
        test_data["train"].to_csv(train_file, index=False)
        test_data["test"].to_csv(test_file, index=False)

        # Mock config
        with patch("training.config") as mock_config:
            mock_config.MODEL_NAME = "test-model"

            with ExitStack() as stack:
                mock_set_uri = stack.enter_context(patch("training.mlflow.set_tracking_uri"))
                mock_set_exp = stack.enter_context(patch("training.mlflow.set_experiment"))
                stack.enter_context(patch("training.mlflow.start_run"))
                stack.enter_context(patch("training.mlflow.log_params"))
                stack.enter_context(patch("training.mlflow.log_metrics"))
                stack.enter_context(patch("training.mlflow.sklearn.log_model"))
                stack.enter_context(patch("training.infer_signature"))
                stack.enter_context(patch("training.joblib.dump"))
                mock_path = stack.enter_context(patch("training.Path"))
                mock_read_csv = stack.enter_context(patch("training.pd.read_csv"))
                mock_rf_class = stack.enter_context(patch("training.RandomForestClassifier"))

                mock_path.return_value.mkdir.return_value = None
                mock_read_csv.side_effect = [test_data["train"], test_data["test"]]

                mock_rf = MagicMock()
                mock_rf.predict.side_effect = [
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                    np.array(["SEKER", "DERMASON", "CALI"]),
                    np.array(["DERMASON", "SEKER", "DERMASON", "CALI", "SEKER", "DERMASON", "CALI"]),
                ]
                mock_rf_class.return_value = mock_rf

                train(
                    train_path=str(train_file),
                    test_path=str(test_file),
                    params={"n_estimators": 10, "min_samples_leaf": 3},
                    model_dir=str(model_dir),
                    register_model=False,
                )

                # Verify MLflow was configured correctly
                mock_set_uri.assert_called_once()
                mock_set_exp.assert_called_once_with("test-model")
