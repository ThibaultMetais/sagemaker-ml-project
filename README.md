# Sagemaker ML Project

A simplified project structure for training and deploying machine learning models with Amazon SageMaker.

## Configuration

This project uses environment variables to manage configuration values, keeping sensitive information out of the source code. 

### Setting up Environment Variables

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit the `.env` file with your actual values. **The following variables are REQUIRED**:
```bash
# AWS Configuration (REQUIRED)
AWS_REGION=us-east-1

# S3 Configuration (REQUIRED)
S3_BUCKET_NAME=your-s3-bucket-name
DEPLOYMENT_BUCKET_PREFIX=sagemaker-models

# ECR Configuration
ECR_REPO_NAME=sagemaker/training_images

# SageMaker Configuration (REQUIRED)
SAGEMAKER_ROLE_ARN=arn:aws:iam::your-account-id:role/your-sagemaker-role
MLFLOW_TRACKING_URI=arn:aws:sagemaker:your-region:your-account-id:mlflow-tracking-server/your-mlflow-server

# Model Configuration
MODEL_NAME=my-example-project

# Instance Configuration
DEFAULT_TRAINING_INSTANCE_TYPE=ml.c5.xlarge
DEFAULT_INFERENCE_INSTANCE_TYPE=ml.m5.large
DEFAULT_INSTANCE_COUNT=1

# Hyperparameters
N_ESTIMATORS=100
MIN_SAMPLES_LEAF=3
```

3. The `.env` file is automatically loaded by the `config.py` module using `python-dotenv`.

### Required Environment Variables

The following environment variables **must** be set in your `.env` file:

- `AWS_ACCOUNT_ID`: Your AWS account ID
- `S3_BUCKET_NAME`: Your S3 bucket name for storing data and models
- `SAGEMAKER_ROLE_ARN`: The ARN of your SageMaker execution role
- `MLFLOW_TRACKING_URI`: The ARN of your MLflow tracking server

If any of these required variables are missing, the application will raise a clear error message.

### Environment-Specific Configuration

The configuration system supports different environments (dev, staging, production). You can specify the environment when running scripts:

```bash
python deploy.py --env production
```

### Security Notes

- The `.env` file is already in `.gitignore` to prevent committing sensitive values
- Never commit actual AWS credentials or sensitive configuration to version control
- Use AWS IAM roles and temporary credentials when possible
- All sensitive values are now required environment variables with no hardcoded defaults

## Project Structure

This project has been reorganized into two main components:

1. ??? : Model-specific code for an XGBoost regression model
2. **sagemaker_utils**: Reusable SageMaker utilities that can be transferred to other projects

### Directory Structure

```
.
├──  ???/                     # Model-specific code
│   ├── training/                  # Training code for the XGBoost model
│   │   ├── hyperparameters.py     # XGBoost hyperparameter configurations
│   │   ├── train.py               # SageMaker training script
│   │   └── evaluation.py          # Model evaluation script
│   └── inference/                 # Inference code for the XGBoost model
│       └── inference.py           # SageMaker inference script
│
├── sagemaker_utils/               # Reusable SageMaker utilities
│   ├── experiments/               # Experiment management
│   │   └── workflow.py            # SageMaker pipelines and workflows
│   ├── jobs/                      # Job management
│   │   └── training_job.py        # Training job utilities
│   ├── registry/                  # Model registry
│   │   └── model_registry.py      # Model registration and versioning
│   └── deployment/                # Deployment utilities
│       └── endpoint.py            # Endpoint management
│
├── examples/                      # Example scripts
│   └── train_and_deploy.py        # Example script for training and deployment
│
└── ...
```

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -e .
```

3. Configure AWS credentials and environment variables:

```bash
cp env.example .env
# Edit .env with your credentials and configuration
```

## Usage

### Example: Train and Deploy a Model

```bash
python examples/train_and_deploy.py \
  --environment dev \
  --model-name xgboost-model \
  --train-data s3://your-bucket/train/ \
  --validation-data s3://your-bucket/validation/ \
  --output-path s3://your-bucket/models/ \
  --role-arn arn:aws:iam::123456789012:role/SageMakerExecutionRole \
  --deploy
```

## Project Modules

### ???

Contains model-specific code for your XGBoost model:

- **training/hyperparameters.py**: Defines the XGBoost hyperparameters
- **training/train.py**: The SageMaker training script for model training
- **training/evaluation.py**: Script for evaluating model performance
- **inference/inference.py**: SageMaker inference script for model serving

### sagemaker_utils

Contains reusable SageMaker utilities that can be used across projects:

- **experiments**: Experiment management with SageMaker pipelines
- **jobs**: Training job management
- **registry**: Model registry management
- **deployment**: Endpoint deployment and management

## Development

### Adding New Models

To add a new model, create a new module in the `???` directory with the appropriate training code and hyperparameters.

### Contributing to SageMaker Utilities

The `sagemaker_utils` package is designed to be reusable across projects. Consider extracting it to a separate repository or Python package in the future.
