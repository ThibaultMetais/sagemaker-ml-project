"""
Model deployment module for the SageMaker + MLflow project.

This module handles the complete deployment pipeline for MLflow models to
SageMaker endpoints. It includes functionality for ECR repository management,
Docker image building, and SageMaker endpoint deployment.

The module bridges the gap between MLflow model registry and SageMaker
deployment by building custom Docker images that can serve MLflow models
on SageMaker endpoints, even though SageMaker doesn't natively support
MLflow inference images.
"""

import argparse
import subprocess
from pathlib import Path

import boto3
import docker
import mlflow
import mlflow.deployments
from botocore.exceptions import ClientError
from mlflow.tracking import MlflowClient

from config import Config


def ensure_ecr_repo_exists(ecr_client, repo_name: str):
    """
    Ensure that an ECR repository exists, creating it if necessary.

    This function checks if the specified ECR repository exists and creates
    it if it doesn't. This is necessary before pushing Docker images to ECR.

    Args:
        ecr_client: Boto3 ECR client instance.
        repo_name (str): Name of the ECR repository to check/create.

    Returns:
        None

    Raises:
        ClientError: If there's an error with the ECR API call (other than
                    repository not found).
    """
    try:
        # Check if the repository exists
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        print(f"ECR repository '{repo_name}' exists.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            # Repository doesn't exist, create it
            print(f"ECR repository '{repo_name}' not found. Creating it...")
            ecr_client.create_repository(repositoryName=repo_name)
        else:
            # Re-raise other errors
            raise


def image_exists_in_ecr(ecr_client, repo_name: str, tag: str) -> bool:
    """
    Check if a Docker image with the specified tag exists in ECR.

    This function checks if an image with the given tag already exists in
    the specified ECR repository. This is useful to avoid rebuilding and
    pushing images that already exist.

    Args:
        ecr_client: Boto3 ECR client instance.
        repo_name (str): Name of the ECR repository.
        tag (str): Docker image tag to check for.

    Returns:
        bool: True if the image exists, False otherwise.

    Raises:
        ClientError: If there's an error with the ECR API call (other than
                    image not found).
    """
    try:
        # Check if the image exists in the repository
        response = ecr_client.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}])
        return len(response["imageDetails"]) > 0
    except ClientError as e:
        if e.response["Error"]["Code"] == "ImageNotFoundException":
            # Image doesn't exist
            return False
        # Re-raise other errors
        raise


def docker_login_to_ecr(aws_account_id: str, region: str):
    """
    Authenticate Docker with ECR to enable pushing images.

    This function uses the AWS CLI to get ECR login credentials and
    authenticates Docker with the ECR registry. This is required before
    pushing Docker images to ECR.

    Args:
        aws_account_id (str): AWS account ID.
        region (str): AWS region.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the AWS CLI or Docker login command fails.
    """
    # Construct the ECR registry URI
    ecr_uri = f"{aws_account_id}.dkr.ecr.{region}.amazonaws.com"

    # Get ECR login password and authenticate Docker
    # This command pipes the ECR password to Docker login
    subprocess.run(
        f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {ecr_uri}",
        shell=True,
        check=True,
    )


def get_model_uri_from_semver(registered_model_name, desired_semver):
    """
    Get the MLflow model URI for a model version with a specific semantic version tag.

    This function searches through all versions of a registered model to find
    the one with the specified semantic version tag. This allows for precise
    model version selection during deployment.

    Args:
        registered_model_name (str): Name of the registered model in MLflow.
        desired_semver (str): Semantic version tag to search for.

    Returns:
        str: MLflow model URI in the format 'models:/model_name/version'.

    Raises:
        Exception: If no model version is found with the specified semantic version tag.
    """
    client = MlflowClient()
    model_uri = None

    # Search for all versions of the model
    results = client.search_model_versions(f"name='{registered_model_name}'")

    # Find the version with the matching semantic version tag
    for version in results:
        # version.tags is a dict of tags set for this model version
        if version.tags.get("semver") == desired_semver:
            # Construct the model URI in MLflow's standard format
            model_uri = f"models:/{registered_model_name}/{version.version}"
            break

    if model_uri is None:
        raise Exception("No model version found with semver tag: " + desired_semver)

    print("Found model URI:", model_uri)
    return model_uri


def build_mlflow_container(model_uri: str, image_name: str):
    """
    Build a Docker container for MLflow model serving.

    This function generates a Dockerfile for the MLflow model and builds
    a Docker image that can serve the model. The generated Dockerfile
    includes all necessary dependencies for model serving.

    Args:
        model_uri (str): MLflow model URI to build container for.
        image_name (str): Name for the Docker image.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the mlflow or docker build command fails.

    Note:
        This function is fragile and seems to be the only way to build
        Docker V2 manifests for MLflow models. The build process creates
        a temporary .deploy directory with the generated Dockerfile.
    """
    # Create deployment directory for generated files
    Path(Path(__file__).parent.parent / ".deploy").mkdir(parents=True, exist_ok=True)

    # Generate Dockerfile for the MLflow model
    # This creates a Dockerfile that can serve the model
    subprocess.run(
        [
            "mlflow",
            "models",
            "generate-dockerfile",
            "-m",
            model_uri,
            "-d",
            str(Path(__file__).parent.parent / ".deploy"),
        ],
        check=True,
    )

    # Build the Docker image
    # Use linux/amd64 platform for SageMaker compatibility
    # Disable provenance and OCI media types for compatibility
    subprocess.run(
        [
            "docker",
            "build",
            "--platform=linux/amd64",
            "--provenance=false",
            "--output",
            "type=image,push=false,oci-mediatypes=false",
            "-t",
            image_name,
            str(Path(__file__).parent.parent / ".deploy"),
        ],
        check=True,
    )


def tag_and_push_image(local_tag: str, full_ecr_uri: str):
    """
    Tag a local Docker image and push it to ECR.

    This function tags a locally built Docker image with the ECR URI
    and pushes it to the ECR repository.

    Args:
        local_tag (str): Local tag of the Docker image.
        full_ecr_uri (str): Full ECR URI for the image (including tag).

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the docker push command fails.
    """
    # Get the Docker client and image
    client = docker.from_env()
    image = client.images.get(local_tag)

    # Tag the image with the ECR URI
    image.tag(full_ecr_uri)

    # Push the tagged image to ECR
    subprocess.run(["docker", "push", full_ecr_uri], check=True)


def deploy_to_sagemaker(
    model_uri: str,
    endpoint_name: str,
    image_url: str,
    role_arn: str,
    bucket: str,
    region: str,
    instance_type: str,
    instance_count: int = 1,
):
    """
    Deploy an MLflow model to a SageMaker endpoint.

    This function uses MLflow's SageMaker deployment client to create
    a SageMaker endpoint that serves the MLflow model using the specified
    Docker image.

    Args:
        model_uri (str): MLflow model URI to deploy.
        endpoint_name (str): Name for the SageMaker endpoint.
        image_url (str): ECR URL of the Docker image to use for serving.
        role_arn (str): SageMaker execution role ARN.
        bucket (str): S3 bucket for model artifacts.
        region (str): AWS region for the endpoint.
        instance_type (str): SageMaker instance type for the endpoint.
        instance_count (int): Number of instances for the endpoint. Defaults to 1.

    Returns:
        None
    """
    # Get the MLflow SageMaker deployment client
    client = mlflow.deployments.get_deploy_client("sagemaker")

    # Create the deployment
    client.create_deployment(
        name=endpoint_name,
        model_uri=model_uri,
        config={
            "execution_role_arn": role_arn,
            "bucket": bucket,
            "image_url": image_url,
            "region_name": region,
            "instance_type": instance_type,
            "instance_count": instance_count,
        },
    )


def main():
    """
    Main deployment function that orchestrates the complete deployment pipeline.

    This function performs the following steps:
    1. Parses command line arguments for deployment configuration
    2. Gets the model URI from MLflow registry using semantic version
    3. Authenticates with ECR
    4. Ensures ECR repository exists
    5. Builds and pushes Docker image if it doesn't exist
    6. Deploys the model to SageMaker endpoint

    Returns:
        None
    """
    # Initialize configuration
    config = Config()

    # Parse command line arguments for deployment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", help="Deployment environment (dev, staging, prod)")
    parser.add_argument(
        "--instance_type",
        type=str,
        default=config.DEFAULT_INFERENCE_INSTANCE_TYPE,
        help="SageMaker instance type for the endpoint",
    )
    parser.add_argument(
        "--instance_count", type=int, default=config.DEFAULT_INSTANCE_COUNT, help="Number of instances for the endpoint"
    )
    args = parser.parse_args()

    # Extract deployment parameters
    env = args.env
    instance_type = args.instance_type
    instance_count = args.instance_count

    # Configure MLflow tracking URI
    mlflow.set_tracking_uri(config.get_mlflow_tracking_uri())

    # Get the model version from version.txt
    model_version = config.get_model_version()

    # Get AWS and deployment configuration
    aws_region = config.AWS_REGION
    aws_role_arn = config.get_sagemaker_role_arn()
    sagemaker_endpoint_name = config.get_endpoint_name(env)
    s3_bucket = config.get_deployment_bucket(env)
    ecr_repo_name = config.MODEL_NAME

    # Derived values for ECR and Docker operations
    aws_account_id = config.AWS_ACCOUNT_ID
    ecr_client = boto3.client("ecr", region_name=aws_region)
    ecr_image_uri = f"{config.ECR_URI}/{ecr_repo_name}:{model_version}"
    local_image_tag = f"{ecr_repo_name}:latest"

    # Execute the deployment workflow
    print("Starting deployment workflow...")

    # Step 1: Get the model URI from MLflow registry
    model_uri = get_model_uri_from_semver(config.MODEL_NAME, model_version)

    # Step 2: Authenticate with ECR
    docker_login_to_ecr(aws_account_id, aws_region)

    # Step 3: Ensure ECR repository exists
    ensure_ecr_repo_exists(ecr_client, ecr_repo_name)

    # Step 4: Check if image already exists in ECR
    if image_exists_in_ecr(ecr_client, ecr_repo_name, model_version):
        print(f"ECR image already exists: {ecr_image_uri} — skipping push.")
    else:
        # Step 5: Build and push Docker image
        print(f"Building and pushing image: {ecr_image_uri}")
        build_mlflow_container(model_uri, ecr_repo_name)
        tag_and_push_image(local_image_tag, ecr_image_uri)

    # Step 6: Deploy to SageMaker endpoint
    deploy_to_sagemaker(
        model_uri,
        sagemaker_endpoint_name,
        ecr_image_uri,
        aws_role_arn,
        s3_bucket,
        aws_region,
        instance_type,
        instance_count,
    )

    print(f"Model deployed to SageMaker endpoint: {sagemaker_endpoint_name}")


if __name__ == "__main__":
    main()
