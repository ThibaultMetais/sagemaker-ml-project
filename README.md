# SageMaker + MLflow Kickstart Project

This repository is a **kickstart template for machine learning projects** that leverage both **Amazon SageMaker** and **MLflow**. It is designed to help you quickly set up robust, production-grade ML workflows, overcoming the current limitations of SageMaker's native MLflow integration.

## Why This Project?

Amazon SageMaker does not natively support MLflow inference images, and its MLflow integration is incomplete. This project bridges that gap, enabling you to:

- **Train any ML model** using SageMaker's managed infrastructure.
- **Perform hyperparameter tuning** with SageMaker's built-in capabilities.
- **Track experiments, register models, and manage the model lifecycle** using MLflow's experiment tracking and model registry.
- **Deploy models for inference** using custom-built images that are compatible with both SageMaker and MLflow, even though SageMaker does not support MLflow inference images out of the box.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **AWS Account** with appropriate permissions for SageMaker, S3, ECR, and IAM
- **MLflow Tracking Server** (self-hosted or managed)
- **Docker** (for building inference images)
- **Git** for version control

---

## Features

- **Seamless integration** of SageMaker training, hyperparameter tuning, and MLflow experiment/model management.
- **Reusable utilities** for SageMaker and MLflow, designed for easy transfer to other projects.
- **Environment-based configuration** using `.env` files and environment variables.
- **Modern Python packaging** and dependency management with [uv](https://github.com/astral-sh/uv).
- **Comprehensive test suite** and code quality tools.
- **Docker-based deployment** for MLflow models on SageMaker endpoints.

---

## Project Structure

```
.
├── src/                          # Main source code
│   ├── __init__.py
│   ├── config.py                 # Configuration management (env, AWS, etc.)
│   ├── prepare_data.py           # Data preparation and preprocessing
│   ├── training.py               # Model training (SageMaker + MLflow)
│   ├── estimator.py              # SageMaker estimator setup
│   ├── hyperparameter_tuning.py  # SageMaker hyperparameter tuning
│   ├── deploy.py                 # Model deployment (SageMaker + MLflow)
│   ├── utils.py                  # Shared utilities
│   └── test_mlflow_inference.py  # MLflow inference testing
├── tests/                        # Comprehensive test suite (pytest)
│   ├── __init__.py
│   ├── conftest.py               # Pytest configuration and fixtures
│   ├── test_config.py            # Configuration tests
│   ├── test_prepare_data.py      # Data preparation tests
│   ├── test_training.py          # Training tests
│   ├── test_estimator.py         # Estimator tests
│   ├── test_hyperparameter_tuning.py  # Hyperparameter tuning tests
│   ├── test_deploy.py            # Deployment tests
│   ├── test_utils.py             # Utilities tests
│   ├── test_mlflow_inference.py  # MLflow inference tests
│   └── README.md                 # Test documentation
├── .circleci/                    # CI/CD configuration
│   └── config.yml                # CircleCI pipeline configuration
├── env.example                   # Example environment variables file
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # UV dependency lock file
├── makefile                      # Build and deployment commands
├── Dockerfile                    # Main Docker image for training
├── build_and_publish.sh          # Script to build and publish Docker images
├── version.txt                   # Project version file
├── requirements_hash.txt         # Requirements hash for rebuild detection
├── .gitignore                    # Git ignore rules
├── .python-version               # Python version specification
└── README.md                     # This file
```

---

## Quickstart

### 1. Install [uv](https://github.com/astral-sh/uv) (Recommended)

```bash
pip install uv
```

### 2. Install and Configure AWS CLI

**Install AWS CLI:**
```bash
# macOS
brew install awscli

# Ubuntu/Debian
sudo apt-get install awscli

# Windows
# Download from https://aws.amazon.com/cli/
```

**Configure AWS credentials:**
```bash
aws configure
```

You'll be prompted for:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., us-east-1)
- Default output format (json)

**Alternative authentication methods:**
- **IAM Roles** (if running on EC2 or ECS)
- **AWS SSO** for enterprise environments
- **Environment variables** (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)

### 3. Set Up Your Environment

```bash
cp env.example .env
# Edit .env with your AWS and MLflow details (see below)
```

**Required variables in `.env`:**

```
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/your-sagemaker-role
MLFLOW_TRACKING_URI=https://mlflow.yourdomain.com
MODEL_NAME=my-ml-model
# ... (see env.example for all options)
```

### 4. Install Dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

Or, if you prefer, use:

```bash
uv sync
```

### 5. Verify AWS Setup

Test your AWS configuration:
```bash
aws sts get-caller-identity
```

This should return your AWS account ID, user ID, and ARN.

---

## Usage

### Data Preparation

```bash
make prepare-data
```

This will:
- Download the UCI Dry Bean dataset
- Perform data preprocessing and visualization
- Split data into train/test sets
- Save processed data locally and to S3

### Local Training

```bash
make train
```

Trains the model locally without registering it in MLflow.

### Local Training with MLflow Registration

```bash
make train-register-model
```

Trains the model locally and registers it in MLflow's model registry.

### SageMaker Training

```bash
make train-sagemaker
```

Trains the model on SageMaker using the configured estimator.

### Hyperparameter Tuning

```bash
make train-sagemaker-hyperparameter-tuning
```

Runs hyperparameter tuning on SageMaker to find optimal model parameters.

### Model Deployment

```bash
make deploy-sagemaker-endpoint
```

Deploys the model to a SageMaker endpoint for inference.

### Testing MLflow Inference

```bash
make test-mlflow-inference
```

Tests the MLflow model inference locally.

### Local Inference Testing

```bash
make inference
```

Tests the inference endpoint locally (requires running endpoint).

### Development Commands

```bash
make test      # Run all tests
make lint      # Check code quality
make format    # Format code
make clean     # Clean generated files
make help      # Show all available commands
```

---

## Environment Variables

All configuration is managed via environment variables. Copy `env.example` to `.env` and configure:

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AWS_ACCOUNT_ID` | Your AWS account ID | `123456789012` |
| `AWS_REGION` | AWS region for resources | `us-east-1` |
| `S3_BUCKET_NAME` | S3 bucket for data and models | `my-ml-bucket` |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN | `arn:aws:iam::123456789012:role/SageMakerExecutionRole` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `https://mlflow.yourdomain.com` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEPLOYMENT_BUCKET_PREFIX` | S3 prefix for deployments | `sagemaker-models` |
| `ECR_REPO_NAME` | ECR repository name | `sagemaker/training_images` |
| `MODEL_NAME` | Model name for registry | `my-example-project` |
| `DEFAULT_TRAINING_INSTANCE_TYPE` | SageMaker training instance | `ml.c5.xlarge` |
| `DEFAULT_INFERENCE_INSTANCE_TYPE` | SageMaker inference instance | `ml.m5.large` |
| `DEFAULT_INSTANCE_COUNT` | Number of inference instances | `1` |
| `N_ESTIMATORS` | XGBoost n_estimators parameter | `100` |
| `MIN_SAMPLES_LEAF` | XGBoost min_samples_leaf parameter | `3` |

**Tip:** The `.env` file is loaded automatically by the project and is already in `.gitignore`.

---

## Development & Testing

### Running Tests

```bash
make test
```

### Code Quality

```bash
make lint      # Check for issues
make format    # Auto-format code
```

### Test Coverage

```bash
pytest --cov=src --cov-report=term-missing --cov-report=html
```

Coverage reports are generated in the `htmlcov/` directory.

---

## AWS Permissions

Your AWS user/role needs the following permissions:

### SageMaker
- `sagemaker:CreateTrainingJob`
- `sagemaker:CreateHyperParameterTuningJob`
- `sagemaker:CreateModel`
- `sagemaker:CreateEndpoint`
- `sagemaker:CreateEndpointConfig`
- `sagemaker:InvokeEndpoint`

### S3
- `s3:GetObject`
- `s3:PutObject`
- `s3:ListBucket`

### ECR
- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`
- `ecr:GetDownloadUrlForLayer`
- `ecr:BatchGetImage`
- `ecr:InitiateLayerUpload`
- `ecr:UploadLayerPart`
- `ecr:CompleteLayerUpload`
- `ecr:PutImage`

### IAM
- `iam:PassRole` (for SageMaker execution role)

---

## Troubleshooting

### Common Issues

**AWS Credentials Not Found**
```bash
aws sts get-caller-identity
# If this fails, run: aws configure
```

**SageMaker Role Permissions**
Ensure your SageMaker execution role has the necessary permissions for S3, ECR, and CloudWatch.

**MLflow Connection Issues**
Verify your MLflow tracking URI is accessible and credentials are correct.

**Docker Build Failures**
Ensure Docker is running and you have sufficient disk space.

### Debug Mode

Set environment variable for verbose logging:
```bash
export LOG_LEVEL=DEBUG
```

---

## Notes & Limitations

- **MLflow Inference Images:** SageMaker does not support MLflow inference images natively. This project provides custom logic to enable MLflow model serving on SageMaker endpoints.
- **Reusable Utilities:** The `src/` directory contains utilities for SageMaker and MLflow that can be reused in other projects.
- **Security:** Never commit your `.env` file or credentials to version control. Use IAM roles and best practices for AWS security.
- **Costs:** Running SageMaker training jobs and endpoints incurs AWS charges. Monitor your usage and clean up resources when not needed.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone <your-fork-url>
cd sagemaker-project
cp env.example .env
# Configure your .env file
make test  # Ensure all tests pass
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Issues:** Use GitHub Issues for bug reports and feature requests
- **Documentation:** Check the `tests/` directory for usage examples
- **Community:** Join our discussions in GitHub Discussions

---

**Kickstart your ML project with the best of SageMaker and MLflow—without the integration headaches!**

*Built with ❤️ for the ML community*
