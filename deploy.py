import argparse
from pathlib import Path
import boto3
import subprocess
import docker
import mlflow
import mlflow.deployments
from mlflow.tracking import MlflowClient
from botocore.exceptions import ClientError

from config import Config


def get_aws_account_id() -> str:
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def ensure_ecr_repo_exists(ecr_client, repo_name: str):
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        print(f"ECR repository '{repo_name}' exists.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            print(f"ECR repository '{repo_name}' not found. Creating it...")
            ecr_client.create_repository(repositoryName=repo_name)
        else:
            raise


def image_exists_in_ecr(ecr_client, repo_name: str, tag: str) -> bool:
    try:
        response = ecr_client.describe_images(
            repositoryName=repo_name, imageIds=[{"imageTag": tag}]
        )
        return len(response["imageDetails"]) > 0
    except ClientError as e:
        if e.response["Error"]["Code"] == "ImageNotFoundException":
            return False
        raise


def docker_login_to_ecr(aws_account_id: str, region: str):
    ecr_uri = f"{aws_account_id}.dkr.ecr.{region}.amazonaws.com"
    subprocess.run(
        f"aws ecr get-login-password --region {region} | "
        f"docker login --username AWS --password-stdin {ecr_uri}",
        shell=True,
        check=True,
    )


def get_model_uri_from_semver(registered_model_name, desired_semver):
    client = MlflowClient()
    model_uri = None

    # Search for all versions of the model.
    results = client.search_model_versions(f"name='{registered_model_name}'")
    for version in results:
        # version.tags is a dict of tags set for this model version
        if version.tags.get("semver") == desired_semver:
            # Construct the model URI in MLflow's standard format.
            model_uri = f"models:/{registered_model_name}/{version.version}"
            break

    if model_uri is None:
        raise Exception("No model version found with semver tag: " + desired_semver)

    print("Found model URI:", model_uri)
    return model_uri


# This is incredibly fragile, and seems to be the only way to build Docker V2 manifests.
def build_mlflow_container(model_uri: str, image_name: str):
    Path(".deploy").mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "mlflow",
            "models",
            "generate-dockerfile",
            "-m",
            model_uri,
            "-d",
            str(Path(__file__).parent / ".deploy"),
        ],
        check=True,
    )

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
            ".deploy/.",
        ],
        check=True,
    )


def tag_and_push_image(local_tag: str, full_ecr_uri: str):
    client = docker.from_env()
    image = client.images.get(local_tag)
    image.tag(full_ecr_uri)
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
    client = mlflow.deployments.get_deploy_client("sagemaker")
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


if __name__ == "__main__":
    # Get configuration
    config = Config()
    
    # get parameters from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev")
    parser.add_argument("--instance_type", type=str, default=config.DEFAULT_INFERENCE_INSTANCE_TYPE)
    parser.add_argument("--instance_count", type=int, default=config.DEFAULT_INSTANCE_COUNT)
    args = parser.parse_args()

    env = args.env
    instance_type = args.instance_type
    instance_count = args.instance_count

    # Set the tracking server URI using the ARN of the tracking server you created
    mlflow.set_tracking_uri(config.get_mlflow_tracking_uri())

    # get the version from version.txt
    model_version = config.get_model_version()

    aws_region = config.AWS_REGION
    aws_role_arn = config.get_sagemaker_role_arn()
    sagemaker_endpoint_name = config.get_endpoint_name(env)
    s3_bucket = config.get_deployment_bucket(env)
    ecr_repo_name = config.MODEL_NAME

    # Derived values
    aws_account_id = get_aws_account_id()
    ecr_client = boto3.client("ecr", region_name=aws_region)
    ecr_image_uri = f"{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{ecr_repo_name}:{model_version}"
    local_image_tag = f"{ecr_repo_name}:latest"

    # Workflow
    model_uri = get_model_uri_from_semver(config.MODEL_NAME, model_version)
    docker_login_to_ecr(aws_account_id, aws_region)
    ensure_ecr_repo_exists(ecr_client, ecr_repo_name)

    if image_exists_in_ecr(ecr_client, ecr_repo_name, model_version):
        print(f"ECR image already exists: {ecr_image_uri} — skipping push.")
    else:
        print(f"Tagging and pushing image: {ecr_image_uri}")
        build_mlflow_container(model_uri, ecr_repo_name)
        tag_and_push_image(local_image_tag, ecr_image_uri)

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
