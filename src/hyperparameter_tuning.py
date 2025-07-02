"""
Hyperparameter tuning module for the SageMaker + MLflow project.

This module provides functionality to perform hyperparameter tuning using
SageMaker's built-in hyperparameter tuning service. It automates the process
of finding optimal hyperparameters for the Random Forest classifier by running
multiple training jobs with different parameter combinations.

The module uses SageMaker's HyperparameterTuner to optimize the model's
performance based on balanced accuracy metrics extracted from training logs.
"""

import time

import sagemaker
from sagemaker.tuner import IntegerParameter

from config import Config
from estimator import build_estimator

# Initialize configuration
config = Config()

# Get S3 data paths for training data
data_paths = config.get_data_paths()

# Define hyperparameter exploration boundaries
# These ranges define the search space for hyperparameter optimization
hyperparameter_ranges = {
    "n-estimators": IntegerParameter(20, 100),  # Number of trees: 20 to 100
    "min-samples-leaf": IntegerParameter(2, 6),  # Min samples per leaf: 2 to 6
}

# Create the SageMaker HyperparameterTuner
# This will orchestrate multiple training jobs to find optimal hyperparameters
Optimizer = sagemaker.tuner.HyperparameterTuner(
    # Base estimator to use for all training jobs
    estimator=build_estimator(register_model=False),  # Don't register models during tuning
    # Hyperparameter ranges to explore
    hyperparameter_ranges=hyperparameter_ranges,
    # Job naming and optimization settings
    base_tuning_job_name=f"{config.MODEL_NAME}-tuner",  # Base name for tuning jobs
    objective_type="Maximize",  # We want to maximize the objective metric
    objective_metric_name="balanced-accuracy",  # Metric to optimize
    # Metric definition for extracting balanced accuracy from training logs
    # This regex pattern extracts the test balanced accuracy from log output
    metric_definitions=[{"Name": "balanced-accuracy", "Regex": "Test balanced accuracy: ([0-9.]+).*$"}],
    # Tuning job configuration
    max_jobs=10,  # Maximum number of training jobs to run
    max_parallel_jobs=2,  # Maximum number of parallel training jobs
)

if __name__ == "__main__":
    # Start the hyperparameter tuning job
    # This will run multiple training jobs with different hyperparameter combinations
    Optimizer.fit(
        {
            "train": data_paths["train_data_path"],  # Training data channel
            "test": data_paths["test_data_path"],  # Test data channel
        }
    )

    # Get tuning results and display them
    # The results dataframe contains metrics for all training jobs
    results = Optimizer.analytics().dataframe()

    # Wait for results to be available
    # Sometimes there's a delay before results are accessible
    while results.empty:
        time.sleep(1)  # Wait 1 second before checking again
        results = Optimizer.analytics().dataframe()

    # Display the top results
    print("Top hyperparameter tuning results:")
    print(results.head())

    # Get the best estimator (the one with the highest balanced accuracy)
    best_estimator = Optimizer.best_estimator()
    print(f"Best estimator: {best_estimator}")
