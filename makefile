# Makefile – sagemaker-ml-project

PYTHON=python
SRC=src
TRAIN_DATA_PATH=data/train
TEST_DATA_PATH=data/test
TRAIN_DATA_FILE=dry-bean-train.csv
TEST_DATA_FILE=dry-bean-test.csv
MODEL_DIR=models

.PHONY: train evaluate serve test lint format inference clean help

## Prepare data
prepare-data:
	$(PYTHON) $(SRC)/prepare_data.py

## Train the model locally without registering it in MLFlow's Model Registry
train:
	$(PYTHON) $(SRC)/training.py --train $(TRAIN_DATA_PATH) --test $(TEST_DATA_PATH) --train-file $(TRAIN_DATA_FILE) --test-file $(TEST_DATA_FILE) --model-dir $(MODEL_DIR)

## Train the model locally and register it in MLFlow's Model Registry
train-register-model:
	$(PYTHON) $(SRC)/training.py --train $(TRAIN_DATA_PATH) --test $(TEST_DATA_PATH) --train-file $(TRAIN_DATA_FILE) --test-file $(TEST_DATA_FILE) --model-dir $(MODEL_DIR) --register-model

## Train the model on sagemaker
train-sagemaker:
	$(PYTHON) $(SRC)/estimator.py

## Train the model on sagemaker with hyperparameter tuning
train-sagemaker-hyperparameter-tuning:
	$(PYTHON) $(SRC)/hyperparameter_tuning.py

## Test the model on the test data
test-mlflow-inference:
	$(PYTHON) $(SRC)/test_mlflow_inference.py

## Deploy the model to Sagemaker Endpoints
deploy-sagemaker-endpoint:
	$(PYTHON) $(SRC)/deploy.py

## Run unit tests
test:
	pytest tests/

## Format code
format:
	ruff format .

## Lint code (only check, no changes)
lint:
	ruff check .

## Test inference endpoint locally
inference:
	curl -X POST http://localhost:8000/inference \
		-H "Content-Type: application/json" \
		-d '{"features": [1, 2, 3, 4]}'

## Clean up generated files
clean:
	rm -rf $(TRAIN_DATA_PATH)
	rm -rf $(TEST_DATA_PATH)
	rm -rf $(MODEL_DIR)

## Show help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^##' Makefile | sed -E 's/## ?//'
