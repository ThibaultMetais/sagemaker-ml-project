import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


def generate_requirements_with_uv(pyproject_path: Path, uv_lock_path: Path):
    # Create a temp directory to work in
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    pyproject_dst = temp_path / "pyproject.toml"
    shutil.copy(pyproject_path, pyproject_dst)

    if Path(uv_lock_path).exists():
        shutil.copy(uv_lock_path, temp_path / "uv.lock")

    # Compile the requirements.txt using uv
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

    # Get the path to the compiled requirements
    req_path = Path(temp_dir) / "requirements.txt"
    return str(req_path)


def rebuild_training_image_if_requirements_changed(
    requirements_path: Path,
    hash_file_path: Path,
    build_script: str = "build_and_publish.sh",
) -> bool:
    """
    Checks if the contents of the given requirements file have changed, and if so,
    runs the provided build script to rebuild the training image.

    Parameters:
        requirements_path (Path): Path to the generated requirements file.
        hash_file_path (Path): File path to store the last known hash of the requirements file.
        build_script (Path): Shell script to run the training image build process.

    Returns:
        bool: True if the training image was rebuilt (i.e. changes detected), False otherwise.
    """
    # Compute the current SHA256 hash of the requirements file.
    try:
        with open(requirements_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        print(f"Error reading {requirements_path}: {e}")
        return False

    # Try to read the previously stored hash.
    try:
        previous_hash = hash_file_path.read_text().strip()
    except FileNotFoundError:
        previous_hash = ""

    # If the current hash is different from the stored one, rebuild the image.
    if current_hash != previous_hash:
        print("Changes detected in requirements. Rebuilding training image...")
        try:
            subprocess.run(["bash", str(build_script)], check=True)
            # After a successful build, update the stored hash.
            hash_file_path.write_text(current_hash)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Build script failed: {e}")
            # Don't update the hash if the build failed
            return False
    else:
        print("No changes detected in requirements. Skipping rebuild.")
        return False
