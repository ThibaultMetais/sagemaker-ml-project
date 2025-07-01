import argparse

import mlflow.sklearn
import pandas as pd

from config import Config

# Get configuration
config = Config()

# Get the model version from version.txt
model_version = config.get_model_version()

# Get data paths
data_paths = config.get_data_paths()


def test_mlflow_inference(run_id: str = None):
    # Set the tracking server URI using the ARN of the tracking server you created
    mlflow.set_tracking_uri(config.get_mlflow_tracking_uri())
    client = mlflow.tracking.MlflowClient()

    # Load model from run or registry
    model = None
    if run_id:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    else:
        # Get model version from registry
        versions = client.search_model_versions(f"name='{config.MODEL_NAME}'")

        # Find version with matching semver tag
        target_version = None
        for version in versions:
            if version.tags.get("semver") == config.get_model_version():
                target_version = version
                break

        if not target_version:
            raise ValueError(
                f"No version of model '{config.MODEL_NAME}' with tag 'semver={config.get_model_version()}' found."
            )

        model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{target_version.version}")

    if not model:
        raise ValueError("Model not loaded successfully")

    # Prepare some sample input (adjust to match your model's expected format)
    df = pd.read_csv(data_paths["test_data_path"] + "/dry-bean-test.csv")
    df = df.drop("Class", axis=1)

    # Run predictions
    predictions = model.predict(df)
    print("Predictions: ", predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, help="Run ID")
    args = parser.parse_args()
    test_mlflow_inference(args.run_id)
