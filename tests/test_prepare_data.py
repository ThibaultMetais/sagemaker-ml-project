"""
Unit tests for the prepare_data.py module.

This module contains comprehensive tests for data preparation functionality,
including UCI dataset fetching, data preprocessing, visualization, and S3 upload.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from prepare_data import prepare_data


class TestPrepareData:
    """Test cases for the prepare_data function."""

    def test_prepare_data_local_only(self, mock_ucimlrepo, mock_matplotlib, temp_files):
        """Test prepare_data function without S3 upload."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }

            # Mock sagemaker.Session
            with patch("prepare_data.sagemaker.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                # Create test data directory
                data_dir = temp_files["temp_dir"] / "data"
                train_dir = data_dir / "train"
                test_dir = data_dir / "test"

                with patch("prepare_data.Path") as mock_path:
                    mock_path.return_value.mkdir.return_value = None

                    # Mock the data directory creation
                    def mock_mkdir(parents=False, exist_ok=False):
                        if str(mock_path.return_value) == "data":
                            data_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/train":
                            train_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/test":
                            test_dir.mkdir(parents=True, exist_ok=True)

                    mock_path.return_value.mkdir.side_effect = mock_mkdir

                    # Mock pandas to_csv and visualization calls
                    with ExitStack() as stack:
                        mock_to_csv = stack.enter_context(patch("pandas.DataFrame.to_csv"))
                        mock_pairplot = stack.enter_context(patch("prepare_data.sns.pairplot"))
                        mock_heatmap = stack.enter_context(patch("prepare_data.sns.heatmap"))
                        mock_show = stack.enter_context(patch("prepare_data.plt.show"))
                        mock_le = stack.enter_context(patch("prepare_data.LabelEncoder"))

                        # Mock label encoder
                        mock_le_instance = MagicMock()
                        mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                        mock_le.return_value = mock_le_instance

                        prepare_data(send_to_s3=False)

                        # Verify pandas to_csv was called
                        mock_to_csv.assert_called()

                        # Verify visualization calls were made
                        mock_pairplot.assert_called()
                        mock_heatmap.assert_called()
                        mock_show.assert_called()

    def test_prepare_data_with_s3_upload(self, mock_ucimlrepo, mock_matplotlib, mock_sagemaker_session, temp_files):
        """Test prepare_data function with S3 upload."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.S3_BUCKET_NAME = "test-bucket"

            # Create test data directory
            data_dir = temp_files["temp_dir"] / "data"
            train_dir = data_dir / "train"
            test_dir = data_dir / "test"

            with patch("prepare_data.Path") as mock_path:
                # Mock Path constructor to return actual paths for specific strings
                def mock_path_constructor(path_str):
                    if path_str == "data":
                        return data_dir
                    elif path_str == "data/train":
                        return train_dir
                    elif path_str == "data/test":
                        return test_dir
                    else:
                        # For any other path, return a mock
                        mock_path_obj = MagicMock()
                        mock_path_obj.mkdir.return_value = None
                        return mock_path_obj

                mock_path.side_effect = mock_path_constructor

                with ExitStack() as stack:
                    stack.enter_context(patch("pandas.DataFrame.to_csv"))
                    stack.enter_context(patch("prepare_data.sns.pairplot"))
                    stack.enter_context(patch("prepare_data.sns.heatmap"))
                    stack.enter_context(patch("prepare_data.plt.show"))
                    mock_le = stack.enter_context(patch("prepare_data.LabelEncoder"))

                    # Mock label encoder
                    mock_le_instance = MagicMock()
                    mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                    mock_le.return_value = mock_le_instance

                    prepare_data(send_to_s3=True)

                    # Verify S3 upload was called
                    assert mock_sagemaker_session.upload_data.call_count == 2

                    # Verify upload calls
                    upload_calls = mock_sagemaker_session.upload_data.call_args_list
                    assert len(upload_calls) == 2

                    # Check train data upload
                    train_call = upload_calls[0]
                    assert train_call[1]["path"] == "data/train/dry-bean-train.csv"
                    assert train_call[1]["bucket"] == "test-bucket"
                    assert "train" in train_call[1]["key_prefix"]

                    # Check test data upload
                    test_call = upload_calls[1]
                    assert test_call[1]["path"] == "data/test/dry-bean-test.csv"
                    assert test_call[1]["bucket"] == "test-bucket"
                    assert "test" in test_call[1]["key_prefix"]

    def test_prepare_data_visualization_calls(self, mock_ucimlrepo, mock_matplotlib, temp_files):
        """Test that visualization functions are called correctly."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }

            # Mock sagemaker.Session
            with patch("prepare_data.sagemaker.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                # Create test data directory
                data_dir = temp_files["temp_dir"] / "data"
                train_dir = data_dir / "train"
                test_dir = data_dir / "test"

                with patch("prepare_data.Path") as mock_path:
                    mock_path.return_value.mkdir.return_value = None

                    # Mock the data directory creation
                    def mock_mkdir(parents=False, exist_ok=False):
                        if str(mock_path.return_value) == "data":
                            data_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/train":
                            train_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/test":
                            test_dir.mkdir(parents=True, exist_ok=True)

                    mock_path.return_value.mkdir.side_effect = mock_mkdir

                    # Mock pandas to_csv and visualization calls
                    with ExitStack() as stack:
                        stack.enter_context(patch("pandas.DataFrame.to_csv"))
                        mock_pairplot = stack.enter_context(patch("prepare_data.sns.pairplot"))
                        mock_heatmap = stack.enter_context(patch("prepare_data.sns.heatmap"))
                        mock_show = stack.enter_context(patch("prepare_data.plt.show"))
                        mock_le = stack.enter_context(patch("prepare_data.LabelEncoder"))

                        # Mock label encoder
                        mock_le_instance = MagicMock()
                        mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                        mock_le.return_value = mock_le_instance

                        prepare_data(send_to_s3=False)

                        # Verify visualization functions were called
                        mock_pairplot.assert_called_once()
                        mock_heatmap.assert_called_once()
                        mock_show.assert_called_once()

    def test_prepare_data_label_encoding(self, mock_ucimlrepo, mock_matplotlib, temp_files):
        """Test that label encoding is applied correctly."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }

            # Mock sagemaker.Session
            with patch("prepare_data.sagemaker.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                # Create test data directory
                data_dir = temp_files["temp_dir"] / "data"
                train_dir = data_dir / "train"
                test_dir = data_dir / "test"

                with patch("prepare_data.Path") as mock_path:
                    mock_path.return_value.mkdir.return_value = None

                    # Mock the data directory creation
                    def mock_mkdir(parents=False, exist_ok=False):
                        if str(mock_path.return_value) == "data":
                            data_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/train":
                            train_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/test":
                            test_dir.mkdir(parents=True, exist_ok=True)

                    mock_path.return_value.mkdir.side_effect = mock_mkdir

                    # Mock pandas to_csv and capture the data being saved
                    saved_data = []

                    def mock_to_csv(path, **kwargs):
                        # Get the DataFrame that's calling to_csv
                        import pandas as pd

                        # Create a simple mock DataFrame for testing
                        mock_df = pd.DataFrame(
                            {
                                "Area": [100, 200, 300, 400, 500],
                                "MajorAxisLength": [10, 20, 30, 40, 50],
                                "MinorAxisLength": [5, 10, 15, 20, 25],
                                "Eccentricity": [0.1, 0.2, 0.3, 0.4, 0.5],
                                "Roundness": [0.8, 0.7, 0.6, 0.5, 0.4],
                                "Class": [0, 1, 0, 2, 1],  # Encoded values
                            }
                        )
                        saved_data.append((path, mock_df))

                    with patch("pandas.DataFrame.to_csv", side_effect=mock_to_csv), patch(
                        "prepare_data.sns.pairplot"
                    ), patch("prepare_data.sns.heatmap"), patch("prepare_data.plt.show"), patch(
                        "prepare_data.LabelEncoder"
                    ) as mock_le:
                        # Mock label encoder
                        mock_le_instance = MagicMock()
                        mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                        mock_le.return_value = mock_le_instance

                        prepare_data(send_to_s3=False)

                        # Verify label encoder was used
                        mock_le.assert_called_once()
                        mock_le_instance.fit_transform.assert_called_once()

                        # Verify data was saved
                        assert len(saved_data) == 2

    def test_prepare_data_train_test_split(self, mock_ucimlrepo, mock_matplotlib, temp_files):
        """Test that train-test split is performed correctly."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }

            # Mock sagemaker.Session
            with patch("prepare_data.sagemaker.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                # Create test data directory
                data_dir = temp_files["temp_dir"] / "data"
                train_dir = data_dir / "train"
                test_dir = data_dir / "test"

                with patch("prepare_data.Path") as mock_path:
                    mock_path.return_value.mkdir.return_value = None

                    # Mock the data directory creation
                    def mock_mkdir(parents=False, exist_ok=False):
                        if str(mock_path.return_value) == "data":
                            data_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/train":
                            train_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/test":
                            test_dir.mkdir(parents=True, exist_ok=True)

                    mock_path.return_value.mkdir.side_effect = mock_mkdir

                    # Actually run the function and check the split sizes
                    with patch("prepare_data.sns.pairplot"), patch("prepare_data.sns.heatmap"), patch(
                        "prepare_data.plt.show"
                    ), patch("prepare_data.LabelEncoder") as mock_le:
                        # Mock label encoder
                        mock_le_instance = MagicMock()
                        mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                        mock_le.return_value = mock_le_instance

                        # Call the function
                        prepare_data(send_to_s3=False)

                        # Load the actual saved files and check their sizes
                        import pandas as pd

                        train_df = pd.read_csv("data/train/dry-bean-train.csv")
                        test_df = pd.read_csv("data/test/dry-bean-test.csv")
                        assert len(train_df) > 0
                        assert len(test_df) > 0
                        assert len(train_df) + len(test_df) == 5

    def test_prepare_data_correlation_analysis(self, mock_ucimlrepo, mock_matplotlib, temp_files):
        """Test that correlation analysis is performed correctly."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }

            # Mock sagemaker.Session
            with patch("prepare_data.sagemaker.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                # Create test data directory
                data_dir = temp_files["temp_dir"] / "data"
                train_dir = data_dir / "train"
                test_dir = data_dir / "test"

                with patch("prepare_data.Path") as mock_path:
                    mock_path.return_value.mkdir.return_value = None

                    # Mock the data directory creation
                    def mock_mkdir(parents=False, exist_ok=False):
                        if str(mock_path.return_value) == "data":
                            data_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/train":
                            train_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/test":
                            test_dir.mkdir(parents=True, exist_ok=True)

                    mock_path.return_value.mkdir.side_effect = mock_mkdir

                    # Mock pandas to_csv and visualization calls
                    with ExitStack() as stack:
                        stack.enter_context(patch("pandas.DataFrame.to_csv"))
                        stack.enter_context(patch("prepare_data.sns.pairplot"))
                        mock_heatmap = stack.enter_context(patch("prepare_data.sns.heatmap"))
                        stack.enter_context(patch("prepare_data.plt.show"))
                        mock_le = stack.enter_context(patch("prepare_data.LabelEncoder"))

                        # Mock label encoder
                        mock_le_instance = MagicMock()
                        mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                        mock_le.return_value = mock_le_instance

                        prepare_data(send_to_s3=False)

                        # Verify correlation was calculated
                        mock_heatmap.assert_called_once()

    def test_prepare_data_s3_upload_error_handling(
        self, mock_ucimlrepo, mock_matplotlib, mock_sagemaker_session, temp_files
    ):
        """Test error handling when S3 upload fails."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }
            mock_config.S3_BUCKET_NAME = "test-bucket"

            # Mock sagemaker.Session to raise an exception
            mock_sagemaker_session.upload_data.side_effect = Exception("S3 upload failed")

            # Create test data directory
            data_dir = temp_files["temp_dir"] / "data"
            train_dir = data_dir / "train"
            test_dir = data_dir / "test"

            with patch("prepare_data.Path") as mock_path:
                mock_path.return_value.mkdir.return_value = None

                # Mock the data directory creation
                def mock_mkdir(parents=False, exist_ok=False):
                    if str(mock_path.return_value) == "data":
                        data_dir.mkdir(parents=True, exist_ok=True)
                    elif str(mock_path.return_value) == "data/train":
                        train_dir.mkdir(parents=True, exist_ok=True)
                    elif str(mock_path.return_value) == "data/test":
                        test_dir.mkdir(parents=True, exist_ok=True)

                mock_path.return_value.mkdir.side_effect = mock_mkdir

                # Mock pandas to_csv and visualization calls
                with ExitStack() as stack:
                    # Define mock_to_csv function
                    def mock_to_csv(path, **kwargs):
                        pass

                    stack.enter_context(patch("pandas.DataFrame.to_csv", side_effect=mock_to_csv))
                    stack.enter_context(patch("prepare_data.sns.pairplot"))
                    stack.enter_context(patch("prepare_data.sns.heatmap"))
                    stack.enter_context(patch("prepare_data.plt.show"))
                    mock_le = stack.enter_context(patch("prepare_data.LabelEncoder"))

                    # Mock label encoder
                    mock_le_instance = MagicMock()
                    mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                    mock_le.return_value = mock_le_instance

                    # Call function and expect exception
                    with pytest.raises(Exception, match="S3 upload failed"):
                        prepare_data(send_to_s3=True)

    def test_prepare_data_directory_creation(self, mock_ucimlrepo, mock_matplotlib, temp_files):
        """Test that data directories are created correctly."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }

            # Mock sagemaker.Session
            with patch("prepare_data.sagemaker.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                # Create test data directory
                data_dir = temp_files["temp_dir"] / "data"
                train_dir = data_dir / "train"
                test_dir = data_dir / "test"

                with patch("prepare_data.Path") as mock_path:
                    # Mock Path constructor to return actual paths for specific strings
                    def mock_path_constructor(path_str):
                        if path_str == "data":
                            return data_dir
                        elif path_str == "data/train":
                            return train_dir
                        elif path_str == "data/test":
                            return test_dir
                        else:
                            # For any other path, return a mock
                            mock_path_obj = MagicMock()
                            mock_path_obj.mkdir.return_value = None
                            return mock_path_obj

                    mock_path.side_effect = mock_path_constructor

                    # Mock pandas to_csv and visualization calls
                    with ExitStack() as stack:
                        stack.enter_context(patch("pandas.DataFrame.to_csv"))
                        stack.enter_context(patch("prepare_data.sns.pairplot"))
                        stack.enter_context(patch("prepare_data.sns.heatmap"))
                        stack.enter_context(patch("prepare_data.plt.show"))
                        mock_le = stack.enter_context(patch("prepare_data.LabelEncoder"))

                        # Mock label encoder
                        mock_le_instance = MagicMock()
                        mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                        mock_le.return_value = mock_le_instance

                        prepare_data(send_to_s3=False)

                        # Verify directories were created
                        assert data_dir.exists()
                        assert train_dir.exists()
                        assert test_dir.exists()

    def test_prepare_data_feature_columns(self, mock_ucimlrepo, mock_matplotlib, temp_files):
        """Test that the correct feature columns are used for visualization."""
        # Mock config
        with patch("prepare_data.config") as mock_config:
            mock_config.get_model_version.return_value = "1.0.0"
            mock_config.get_data_paths.return_value = {
                "data_prefix": "test-model/1.0.0/data",
                "data_path": "s3://test-bucket/test-model/1.0.0/data",
                "train_data_path": "s3://test-bucket/test-model/1.0.0/data/train",
                "test_data_path": "s3://test-bucket/test-model/1.0.0/data/test",
            }

            # Mock sagemaker.Session
            with patch("prepare_data.sagemaker.Session") as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session

                # Create test data directory
                data_dir = temp_files["temp_dir"] / "data"
                train_dir = data_dir / "train"
                test_dir = data_dir / "test"

                with patch("prepare_data.Path") as mock_path:
                    mock_path.return_value.mkdir.return_value = None

                    # Mock the data directory creation
                    def mock_mkdir(parents=False, exist_ok=False):
                        if str(mock_path.return_value) == "data":
                            data_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/train":
                            train_dir.mkdir(parents=True, exist_ok=True)
                        elif str(mock_path.return_value) == "data/test":
                            test_dir.mkdir(parents=True, exist_ok=True)

                    mock_path.return_value.mkdir.side_effect = mock_mkdir

                    # Mock pandas to_csv and visualization calls
                    with ExitStack() as stack:
                        stack.enter_context(patch("pandas.DataFrame.to_csv"))
                        mock_pairplot = stack.enter_context(patch("prepare_data.sns.pairplot"))
                        stack.enter_context(patch("prepare_data.sns.heatmap"))
                        stack.enter_context(patch("prepare_data.plt.show"))
                        mock_le = stack.enter_context(patch("prepare_data.LabelEncoder"))

                        # Mock label encoder
                        mock_le_instance = MagicMock()
                        mock_le_instance.fit_transform.return_value = [0, 1, 0, 2, 1]
                        mock_le.return_value = mock_le_instance

                        prepare_data(send_to_s3=False)

                        # Verify pairplot was called with correct feature columns
                        mock_pairplot.assert_called_once()
                        call_args = mock_pairplot.call_args
                        assert "vars" in call_args[1]
                        expected_vars = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "Roundness"]
                        assert call_args[1]["vars"] == expected_vars
