import time
import sagemaker
from sagemaker.tuner import IntegerParameter

from config import Config
from estimator import sklearn_estimator

# Get configuration
config = Config()

# Get data paths
data_paths = config.get_data_paths()

# Define exploration boundaries
hyperparameter_ranges = {
    "n-estimators": IntegerParameter(20, 100),
    "min-samples-leaf": IntegerParameter(2, 6),
}

# Create Optimizer
Optimizer = sagemaker.tuner.HyperparameterTuner(
    estimator=sklearn_estimator,
    hyperparameter_ranges=hyperparameter_ranges,
    base_tuning_job_name=f"{config.MODEL_NAME}-tuner",
    objective_type="Maximize",
    objective_metric_name="balanced-accuracy",
    metric_definitions=[
        {"Name": "balanced-accuracy", "Regex": "Test balanced accuracy: ([0-9.]+).*$"}
    ],  # Extract tracked metric from logs with regexp
    max_jobs=10,
    max_parallel_jobs=2,
)

if __name__ == "__main__":
    Optimizer.fit({"train": data_paths["train_data_path"], "test": data_paths["test_data_path"]})

    # Get tuner results in a df
    results = Optimizer.analytics().dataframe()

    while results.empty:
        time.sleep(1)
        results = Optimizer.analytics().dataframe()
    print(results.head())

    best_estimator = Optimizer.best_estimator()
    print(best_estimator)
