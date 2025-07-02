"""
Utility functions for the SageMaker + MLflow project.

This module provides utility functions for dependency management, Docker image
rebuilding, and other common operations used throughout the project. It includes
functions for generating requirements files using uv, detecting changes in
dependencies, and triggering Docker image rebuilds when necessary.

The utilities are designed to work with the project's uv-based dependency
management system and ensure that training images are rebuilt only when
dependencies actually change.
"""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


def generate_requirements_with_uv(pyproject_path: Path, uv_lock_path: Path):
    """
    Generate a requirements.txt file from pyproject.toml using uv.

    This function creates a temporary directory, copies the pyproject.toml and
    uv.lock files, and uses uv to compile a requirements.txt file with all
    dependencies including extras. This is useful for Docker builds that need
    a traditional requirements.txt file.

    Args:
        pyproject_path (Path): Path to the pyproject.toml file.
        uv_lock_path (Path): Path to the uv.lock file.

    Returns:
        str: Path to the generated requirements.txt file.

    Raises:
        subprocess.CalledProcessError: If the uv compile command fails.
        FileNotFoundError: If pyproject.toml doesn't exist.

    Note:
        The function creates a temporary directory to avoid polluting the
        project directory with generated files. The temporary directory
        should be cleaned up by the caller if needed.
    """
    # Create a temporary directory to work in
    # This prevents polluting the project directory with generated files
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    # Copy pyproject.toml to the temporary directory
    pyproject_dst = temp_path / "pyproject.toml"
    shutil.copy(pyproject_path, pyproject_dst)

    # Copy uv.lock to the temporary directory if it exists
    # This ensures the lock file is available for dependency resolution
    if Path(uv_lock_path).exists():
        shutil.copy(uv_lock_path, temp_path / "uv.lock")

    # Compile the requirements.txt using uv
    # The --all-extras flag includes all optional dependencies
    # This ensures the Docker image has all necessary packages
    subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "--all-extras",
            "--output-file",
            "requirements.txt",
            "pyproject.toml",
        ],
        cwd=temp_path,
        check=True,
    )

    # Return the path to the compiled requirements file
    req_path = Path(temp_dir) / "requirements.txt"
    return str(req_path)


def rebuild_training_image_if_requirements_changed(
    requirements_path: Path,
    hash_file_path: Path,
    build_script: str = "build_and_publish.sh",
) -> bool:
    """
    Check if requirements have changed and rebuild the training image if necessary.

    This function compares the SHA256 hash of the current requirements.txt file
    with a previously stored hash. If the hashes differ, it triggers a rebuild
    of the training Docker image using the specified build script. This ensures
    that training images are only rebuilt when dependencies actually change,
    saving time and resources.

    Args:
        requirements_path (Path): Path to the generated requirements.txt file.
        hash_file_path (Path): File path to store the last known hash of the requirements file.
        build_script (str): Shell script to run the training image build process.
                           Defaults to "build_and_publish.sh".

    Returns:
        bool: True if the training image was rebuilt (i.e. changes detected),
              False otherwise.

    Raises:
        subprocess.CalledProcessError: If the build script fails to execute.

    Note:
        The function updates the hash file only after a successful build.
        If the build fails, the hash is not updated, ensuring the rebuild
        will be attempted again on the next run.
    """
    # Compute the current SHA256 hash of the requirements file
    # This provides a reliable way to detect changes in the file content
    try:
        with open(requirements_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        print(f"Error reading {requirements_path}: {e}")
        return False

    # Try to read the previously stored hash
    # If the file doesn't exist, we assume no previous hash (empty string)
    try:
        previous_hash = hash_file_path.read_text().strip()
    except FileNotFoundError:
        previous_hash = ""

    # If the current hash is different from the stored one, rebuild the image
    # This indicates that dependencies have changed and a rebuild is necessary
    if current_hash != previous_hash:
        print("Changes detected in requirements. Rebuilding training image...")
        try:
            # Execute the build script to rebuild the Docker image
            subprocess.run(["bash", str(build_script)], check=True)

            # After a successful build, update the stored hash
            # This prevents unnecessary rebuilds on subsequent runs
            hash_file_path.write_text(current_hash)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Build script failed: {e}")
            # Don't update the hash if the build failed
            # This ensures the rebuild will be attempted again
            return False
    else:
        print("No changes detected in requirements. Skipping rebuild.")
        return False
