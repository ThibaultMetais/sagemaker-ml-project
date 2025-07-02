# Dockerfile for SageMaker Training Container
# 
# This Dockerfile creates a container image for training machine learning models
# on Amazon SageMaker. The image includes all necessary dependencies for the
# SageMaker + MLflow project, including Python packages, build tools, and
# the SageMaker training toolkit.
# 
# The container is designed to work with SageMaker's training infrastructure
# and supports the complete ML workflow from data loading to model training.

# Use Python 3.11 slim image as base for smaller size and security
# The slim variant reduces attack surface while maintaining functionality
FROM python:3.11-slim-buster

# Install build tools and system dependencies required for Python packages
# These are needed for compiling native extensions in some ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \  # C/C++ compiler and build tools
    gcc \              # GNU C compiler
    && rm -rf /var/lib/apt/lists/*  # Clean up package lists to reduce image size

# Install uv (modern Python package installer written in Rust)
# uv provides faster dependency resolution and installation compared to pip
RUN pip install --no-cache-dir uv

# Install AWS SageMaker training toolkit
# This provides the infrastructure for running training jobs on SageMaker
# including data channel handling, model saving, and logging
RUN pip install --no-cache-dir sagemaker-training

# Set up environment variables that SageMaker expects
# SAGEMAKER_SUBMIT_DIRECTORY tells SageMaker where to find the training code
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Set the default entry point for the container
# This tells SageMaker to use the 'train' command when starting the container
ENTRYPOINT ["train"]