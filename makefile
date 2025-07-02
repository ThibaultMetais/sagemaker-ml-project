# Makefile – SageMaker + MLflow Project
# 
# This Makefile provides convenient commands for managing the complete ML workflow
# including data preparation, model training, testing, deployment, and maintenance.
# 
# The workflow supports both local development and SageMaker deployment scenarios,
# with integration to MLflow for experiment tracking and model registry management.

# Python interpreter to use for all commands
PYTHON=python

# Source code directory containing all Python modules
SRC=src

# Data directory paths for training and testing
TRAIN_DATA_PATH=data/train
TEST_DATA_PATH=data/test

# Data file names for the UCI Dry Bean dataset
TRAIN_DATA_FILE=dry-bean-train.csv
TEST_DATA_FILE=dry-bean-test.csv

# Directory for storing trained models locally
MODEL_DIR=models

# Declare all targets as phony to ensure they always run
.PHONY: train evaluate serve test lint format inference clean help

## Prepare data
## Downloads the UCI Dry Bean dataset, performs preprocessing, and saves to local directories
prepare-data:
	$(PYTHON) $(SRC)/prepare_data.py

## Train the model locally without registering it in MLFlow's Model Registry
## This is useful for development and testing without affecting the model registry
train:
	$(PYTHON) $(SRC)/training.py --train $(TRAIN_DATA_PATH) --test $(TEST_DATA_PATH) --train-file $(TRAIN_DATA_FILE) --test-file $(TEST_DATA_FILE) --model-dir $(MODEL_DIR)

## Train the model locally and register it in MLFlow's Model Registry
## This creates a versioned model in the MLflow registry for deployment
train-register-model:
	$(PYTHON) $(SRC)/training.py --train $(TRAIN_DATA_PATH) --test $(TEST_DATA_PATH) --train-file $(TRAIN_DATA_FILE) --test-file $(TEST_DATA_FILE) --model-dir $(MODEL_DIR) --register-model

## Train the model on SageMaker
## Uses SageMaker's managed infrastructure for scalable training
train-sagemaker:
	$(PYTHON) $(SRC)/estimator.py

## Train the model on SageMaker with hyperparameter tuning
## Automatically finds optimal hyperparameters using SageMaker's tuning service
train-sagemaker-hyperparameter-tuning:
	$(PYTHON) $(SRC)/hyperparameter_tuning.py

## Test the model on the test data
## Validates model performance using MLflow inference testing
test-mlflow-inference:
	$(PYTHON) $(SRC)/test_mlflow_inference.py

## Deploy the model to SageMaker Endpoints
## Creates a production-ready endpoint for real-time inference
deploy-sagemaker-endpoint:
	$(PYTHON) $(SRC)/deploy.py

## Run unit tests
## Executes the complete test suite to ensure code quality
test:
	pytest tests/

## Format code
## Automatically formats all Python code using ruff
format:
	ruff format .

## Lint code (only check, no changes)
## Checks code quality and style without making changes
lint:
	ruff check .

## Test inference endpoint locally
## Sends a test request to a local inference endpoint (requires running endpoint)
inference:
	curl -X POST http://localhost:8000/inference \
		-H "Content-Type: application/json" \
		-d '{"features": [1, 2, 3, 4]}'

## Clean up generated files
## Removes all generated data, models, and temporary files
clean:
	rm -rf $(TRAIN_DATA_PATH)
	rm -rf $(TEST_DATA_PATH)
	rm -rf $(MODEL_DIR)

## Show help
## Displays all available make targets with descriptions
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^##' Makefile | sed -E 's/## ?//'
