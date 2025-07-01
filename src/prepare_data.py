from pathlib import Path

import matplotlib.pyplot as plt
import sagemaker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from config import Config

# Get configuration
config = Config()

# Get the model version from version.txt
model_version = config.get_model_version()

# Get data paths
data_paths = config.get_data_paths()


def prepare_data(send_to_s3: bool = False):
    sess = sagemaker.Session()

    dry_bean = fetch_ucirepo(id=602)["data"]["original"]
    dry_bean.head()

    sns.pairplot(
        dry_bean,
        vars=[
            "Area",
            "MajorAxisLength",
            "MinorAxisLength",
            "Eccentricity",
            "Roundness",
        ],
        hue="Class",
    )

    correlation = dry_bean.corr(numeric_only=True)

    # Create a square heatmap with center at 0
    sns.heatmap(correlation, center=0, square=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.show()

    # For preprocessing
    df = dry_bean.copy(deep=True)

    # Encode the target
    le = LabelEncoder()
    df["Class"] = le.fit_transform(df["Class"])

    # Split the data into two sets
    train, test = train_test_split(df, random_state=1, test_size=0.2)

    # Create data directory if it doesn't exist
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/test").mkdir(parents=True, exist_ok=True)

    train.to_csv("data/train/dry-bean-train.csv")
    test.to_csv("data/test/dry-bean-test.csv")

    if send_to_s3:
        # Send data to S3. SageMaker will take training data from s3
        trainpath = sess.upload_data(
            path="data/train/dry-bean-train.csv",
            bucket=config.S3_BUCKET_NAME,
            key_prefix=f"{data_paths['data_prefix']}/train",
        )

        testpath = sess.upload_data(
            path="data/test/dry-bean-test.csv",
            bucket=config.S3_BUCKET_NAME,
            key_prefix=f"{data_paths['data_prefix']}/test",
        )

        print(f"Train data sent to {trainpath}")
        print(f"Test data sent to {testpath}")


if __name__ == "__main__":
    prepare_data(send_to_s3=True)
