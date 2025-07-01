import argparse
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from config import Config

# Get configuration
config = Config()

# Model name
REGISTERED_MODEL_NAME = config.MODEL_NAME
MLFLOW_TRACKING_URI = config.get_mlflow_tracking_uri()
# get the version from version.txt
model_version = config.get_model_version()


def train(train_path, test_path, params, model_dir, register_model: bool = False):
    print("Reading data")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Building training and testing datasets")
    X_train = train_df.drop("Class", axis=1)
    X_test = test_df.drop("Class", axis=1)
    y_train = train_df["Class"]
    y_test = test_df["Class"]

    print("Training model")
    # Set MLflow tracking URI and experiment settings.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{config.MODEL_NAME}")

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        min_samples_leaf=params["min_samples_leaf"],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Validating model")
    bal_acc_train = balanced_accuracy_score(y_train, model.predict(X_train))
    bal_acc_test = balanced_accuracy_score(y_test, model.predict(X_test))

    print(f"Train balanced accuracy: {bal_acc_train:.3f}")
    print(f"Test balanced accuracy: {bal_acc_test:.3f}")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics({"bal_acc_train": bal_acc_train, "bal_acc_test": bal_acc_test})

        # Infer the model signature for later model serving.
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model WITHOUT automatic registration.
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train,
        )

        if register_model:
            # Now explicitly register the model. This call registers it once and returns a ModelVersion object.
            registered_model = mlflow.register_model(logged_model.model_uri, REGISTERED_MODEL_NAME)

            # Set your custom semantic version tag.
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=registered_model.version,
                key="semver",
                value=model_version,
            )

            print(f"Model registered with version {registered_model.version}, tagged with {model_version}")

    # Save the model to the model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir + "/model.pkl")


if __name__ == "__main__":
    print("Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=3)

    # Data, model, and output directories.
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="dry-bean-train.csv")
    parser.add_argument("--test-file", type=str, default="dry-bean-test.csv")

    # Optional flag to register the model in SageMaker Registry.
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register model in SageMaker Model Registry",
    )
    args, _ = parser.parse_known_args()

    register_model = args.register_model

    params = {
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
    }

    train(
        os.path.join(args.train, args.train_file),
        os.path.join(args.test, args.test_file),
        params,
        args.model_dir,
        register_model=register_model,
    )
