"""
Unit tests for the utils.py module.

This module contains comprehensive tests for utility functions including
requirements generation, hash checking, and build script execution.
"""

import hashlib
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from utils import generate_requirements_with_uv, rebuild_training_image_if_requirements_changed


class TestGenerateRequirementsWithUv:
    """Test cases for the generate_requirements_with_uv function."""

    def test_generate_requirements_with_uv_success(self, temp_files, mock_subprocess):
        """Test successful requirements generation with uv."""
        pyproject_path = temp_files["pyproject_file"]
        uv_lock_path = temp_files["uv_lock_file"]

        # Mock the subprocess.run to return success
        mock_subprocess.return_value.returncode = 0

        result = generate_requirements_with_uv(pyproject_path, uv_lock_path)

        # Verify subprocess was called with correct arguments
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == [
            "uv",
            "pip",
            "compile",
            "--all-extras",
            "--output-file",
            "requirements.txt",
            "pyproject.toml",
        ]
        assert call_args[1]["check"] is True

        # Verify result is a string path
        assert isinstance(result, str)
        assert result.endswith("requirements.txt")

    def test_generate_requirements_with_uv_without_lock_file(self, temp_files, mock_subprocess):
        """Test requirements generation when uv.lock file doesn't exist."""
        pyproject_path = temp_files["pyproject_file"]
        uv_lock_path = Path("nonexistent.lock")

        # Mock the subprocess.run to return success
        mock_subprocess.return_value.returncode = 0

        result = generate_requirements_with_uv(pyproject_path, uv_lock_path)

        # Verify subprocess was called
        mock_subprocess.assert_called_once()

        # Verify result is a string path
        assert isinstance(result, str)
        assert result.endswith("requirements.txt")

    def test_generate_requirements_with_uv_subprocess_failure(self, temp_files, mock_subprocess):
        """Test requirements generation when subprocess fails."""
        pyproject_path = temp_files["pyproject_file"]
        uv_lock_path = temp_files["uv_lock_file"]

        # Mock subprocess to raise CalledProcessError
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "uv")

        with pytest.raises(subprocess.CalledProcessError):
            generate_requirements_with_uv(pyproject_path, uv_lock_path)

    def test_generate_requirements_with_uv_temp_directory_cleanup(self, temp_files, mock_subprocess):
        """Test that temporary directory is properly cleaned up."""
        pyproject_path = temp_files["pyproject_file"]
        uv_lock_path = temp_files["uv_lock_file"]

        # Mock the subprocess.run to return success
        mock_subprocess.return_value.returncode = 0

        # Track the temp directory
        temp_dirs = []
        original_mkdtemp = tempfile.mkdtemp

        def mock_mkdtemp():
            temp_dir = original_mkdtemp()
            temp_dirs.append(temp_dir)
            return temp_dir

        with patch("tempfile.mkdtemp", side_effect=mock_mkdtemp):
            result = generate_requirements_with_uv(pyproject_path, uv_lock_path)

        # Verify temp directory was created and result points to it
        assert len(temp_dirs) == 1
        assert result.startswith(temp_dirs[0])


class TestRebuildTrainingImageIfRequirementsChanged:
    """Test cases for the rebuild_training_image_if_requirements_changed function."""

    def test_rebuild_when_hash_changed(self, temp_files, mock_subprocess):
        """Test that rebuild occurs when requirements hash has changed."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["hash_file"]

        # Write a different hash to the hash file
        hash_file_path.write_text("old_hash")

        # Mock subprocess to return success
        mock_subprocess.return_value.returncode = 0

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "test_build.sh")

        # Verify rebuild occurred
        assert result is True

        # Verify subprocess was called with build script
        mock_subprocess.assert_called_once_with(["bash", "test_build.sh"], check=True)

        # Verify hash was updated
        current_hash = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
        assert hash_file_path.read_text().strip() == current_hash

    def test_no_rebuild_when_hash_unchanged(self, temp_files, mock_subprocess):
        """Test that no rebuild occurs when requirements hash hasn't changed."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["hash_file"]

        # Calculate current hash and write it to hash file
        current_hash = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
        hash_file_path.write_text(current_hash)

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "test_build.sh")

        # Verify no rebuild occurred
        assert result is False

        # Verify subprocess was not called
        mock_subprocess.assert_not_called()

    def test_rebuild_when_hash_file_missing(self, temp_files, mock_subprocess):
        """Test that rebuild occurs when hash file doesn't exist."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["temp_dir"] / "nonexistent_hash.txt"

        # Ensure the hash file doesn't exist
        if hash_file_path.exists():
            hash_file_path.unlink()

        # Mock subprocess to return success
        mock_subprocess.return_value.returncode = 0

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "test_build.sh")

        # Verify rebuild occurred
        assert result is True

        # Verify subprocess was called
        mock_subprocess.assert_called_once()

        # Verify hash file was created
        assert hash_file_path.exists()
        current_hash = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
        assert hash_file_path.read_text().strip() == current_hash

    def test_rebuild_with_custom_build_script(self, temp_files, mock_subprocess):
        """Test that custom build script is used when provided."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["hash_file"]

        # Write a different hash to trigger rebuild
        hash_file_path.write_text("old_hash")

        # Mock subprocess to return success
        mock_subprocess.return_value.returncode = 0

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "custom_build.sh")

        # Verify rebuild occurred
        assert result is True

        # Verify custom build script was used
        mock_subprocess.assert_called_once_with(["bash", "custom_build.sh"], check=True)

    def test_rebuild_with_default_build_script(self, temp_files, mock_subprocess):
        """Test that default build script is used when not provided."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["hash_file"]

        # Write a different hash to trigger rebuild
        hash_file_path.write_text("old_hash")

        # Mock subprocess to return success
        mock_subprocess.return_value.returncode = 0

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path)

        # Verify rebuild occurred
        assert result is True

        # Verify default build script was used
        mock_subprocess.assert_called_once_with(["bash", "build_and_publish.sh"], check=True)

    def test_rebuild_with_requirements_file_error(self, temp_files, mock_subprocess):
        """Test behavior when requirements file cannot be read."""
        requirements_path = Path("nonexistent_requirements.txt")
        hash_file_path = temp_files["hash_file"]

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "test_build.sh")

        # Verify no rebuild occurred due to error
        assert result is False

        # Verify subprocess was not called
        mock_subprocess.assert_not_called()

    def test_rebuild_with_build_script_failure(self, temp_files, mock_subprocess):
        """Test behavior when build script fails."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["hash_file"]

        # Write a different hash to trigger rebuild
        hash_file_path.write_text("old_hash")

        # Mock subprocess to raise CalledProcessError
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "bash")

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "test_build.sh")

        # Verify rebuild failed
        assert result is False

        # Verify subprocess was called
        mock_subprocess.assert_called_once()

        # Verify hash was not updated
        assert hash_file_path.read_text().strip() == "old_hash"

    def test_rebuild_with_empty_requirements_file(self, temp_files, mock_subprocess):
        """Test behavior with empty requirements file."""
        # Create empty requirements file
        empty_requirements = temp_files["temp_dir"] / "empty_requirements.txt"
        empty_requirements.write_text("")

        hash_file_path = temp_files["hash_file"]

        # Mock subprocess to return success
        mock_subprocess.return_value.returncode = 0

        result = rebuild_training_image_if_requirements_changed(empty_requirements, hash_file_path, "test_build.sh")

        # Verify rebuild occurred (empty file has different hash)
        assert result is True

        # Verify subprocess was called
        mock_subprocess.assert_called_once()

        # Verify hash was updated
        empty_hash = hashlib.sha256(b"").hexdigest()
        assert hash_file_path.read_text().strip() == empty_hash

    def test_rebuild_with_large_requirements_file(self, temp_files, mock_subprocess):
        """Test behavior with large requirements file."""
        # Create large requirements file
        large_requirements = temp_files["temp_dir"] / "large_requirements.txt"
        large_content = "pandas>=1.5.0\n" * 1000  # Large content
        large_requirements.write_text(large_content)

        hash_file_path = temp_files["hash_file"]

        # Mock subprocess to return success
        mock_subprocess.return_value.returncode = 0

        result = rebuild_training_image_if_requirements_changed(large_requirements, hash_file_path, "test_build.sh")

        # Verify rebuild occurred
        assert result is True

        # Verify subprocess was called
        mock_subprocess.assert_called_once()

        # Verify hash was updated correctly
        large_hash = hashlib.sha256(large_content.encode()).hexdigest()
        assert hash_file_path.read_text().strip() == large_hash

    def test_rebuild_hash_file_permissions(self, temp_files, mock_subprocess):
        """Test that hash file is written with correct permissions."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["hash_file"]

        # Write a different hash to trigger rebuild
        hash_file_path.write_text("old_hash")

        # Mock subprocess to return success
        mock_subprocess.return_value.returncode = 0

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "test_build.sh")

        # Verify rebuild occurred
        assert result is True

        # Verify hash file is readable
        assert hash_file_path.is_file()
        assert hash_file_path.stat().st_mode & 0o777 == 0o644  # Default file permissions

    def test_rebuild_concurrent_access(self, temp_files, mock_subprocess):
        """Test behavior under concurrent access scenarios."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["hash_file"]

        # Mock subprocess to return success
        mock_subprocess.return_value.returncode = 0

        # Simulate concurrent access by calling function multiple times
        # Each call should see the old hash and trigger a rebuild
        results = []
        for _ in range(3):
            # Reset hash file to old value before each call
            hash_file_path.write_text("old_hash")

            result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "test_build.sh")
            results.append(result)

        # All calls should return True (rebuild occurred)
        assert all(results)

        # Subprocess should be called multiple times
        assert mock_subprocess.call_count == 3

        # Hash should be updated to current value
        current_hash = hashlib.sha256(requirements_path.read_bytes()).hexdigest()
        assert hash_file_path.read_text().strip() == current_hash

    def test_rebuild_with_s3_upload_error(self, temp_files, mock_subprocess):
        """Test behavior when S3 upload fails."""
        requirements_path = temp_files["requirements_file"]
        hash_file_path = temp_files["hash_file"]

        # Set the hash file to old_hash to match test expectation
        hash_file_path.write_text("old_hash")

        # Mock subprocess to raise CalledProcessError with a specific error message
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "bash", stderr=b"S3 upload failed")

        result = rebuild_training_image_if_requirements_changed(requirements_path, hash_file_path, "test_build.sh")

        # Verify rebuild failed
        assert result is False

        # Verify subprocess was called
        mock_subprocess.assert_called_once()

        # Verify hash was not updated
        assert hash_file_path.read_text().strip() == "old_hash"
