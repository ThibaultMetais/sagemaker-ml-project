FROM python:3.11-slim-buster

# Install build tools and lib dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pip installer in Rust) and AWS SageMaker training toolkit
RUN pip install --no-cache-dir uv
RUN pip install --no-cache-dir sagemaker-training

# Set up environment variables SageMaker expects
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENTRYPOINT ["train"]