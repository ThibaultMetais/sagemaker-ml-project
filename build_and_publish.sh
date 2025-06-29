#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Return the exit status of the last command in a pipe that failed

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for required environment variables
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: AWS_ACCOUNT_ID environment variable is required"
    exit 1
fi

if [ -z "$AWS_REGION" ]; then
    echo "Error: AWS_REGION environment variable is required"
    exit 1
fi

if [ -z "$ECR_REPO_NAME" ]; then
    echo "Error: ECR_REPO_NAME environment variable is required"
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME environment variable is required"
    exit 1
fi

IMAGE_NAME="$MODEL_NAME"
# Get the tag from version.txt
TAG=$(cat version.txt)
ECR_REPO="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

echo "Building and publishing image: $ECR_REPO:$IMAGE_NAME"

# Remove any existing builder named sagemaker-builder
docker buildx rm sagemaker-builder 2>/dev/null || true

# Create and use a new builder instance, then bootstrap it
docker buildx create --use --name sagemaker-builder
docker buildx inspect sagemaker-builder --bootstrap

# Build for SageMaker's architecture and load the image locally using the target tag
echo "Building Docker image..."
docker buildx build --platform linux/amd64 -t $ECR_REPO:$IMAGE_NAME --load .

# Authenticate with ECR
echo "Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Push the image to ECR (image is already tagged correctly)
echo "Pushing image to ECR..."
docker push $ECR_REPO:$IMAGE_NAME

echo "Successfully built and published: $ECR_REPO:$IMAGE_NAME"