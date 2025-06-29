import mlflow.sklearn
import pandas as pd

from config import Config

# Get configuration
config = Config()

# Get the model version from version.txt
model_version = config.get_model_version()

# Get data paths
data_paths = config.get_data_paths()

# Set the tracking server URI using the ARN of the tracking server you created
mlflow.set_tracking_uri(config.get_mlflow_tracking_uri())

# Option A: Load from a specific run
# model_uri = "runs:/<your_run_id>/model"  # Replace <your_run_id> with the actual run ID
# model = mlflow.sklearn.load_model(model_uri)

# Option B: Load from the model registry
model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{model_version}")

# Prepare some sample input (adjust to match your model's expected format)
df = pd.read_csv(data_paths["test_data_path"] + "/dry-bean-test.csv")
df = df.drop("Class", axis=1)

# Run predictions
predictions = model.predict(df)
print(predictions)
