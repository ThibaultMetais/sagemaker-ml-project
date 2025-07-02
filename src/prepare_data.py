"""
Data preparation module for the SageMaker + MLflow project.

This module handles the complete data preparation pipeline for the UCI Dry Bean dataset,
including data download, preprocessing, visualization, and storage. It provides
functionality to prepare data for both local development and SageMaker training.

The module downloads the UCI Dry Bean dataset, performs exploratory data analysis,
encodes categorical variables, splits the data into train/test sets, and optionally
uploads the processed data to S3 for SageMaker training.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import sagemaker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from config import Config

# Initialize configuration
config = Config()

# Get the model version from version.txt for data organization
model_version = config.get_model_version()

# Get S3 data paths for organized storage
data_paths = config.get_data_paths()


def prepare_data(send_to_s3: bool = False):
    """
    Prepare the UCI Dry Bean dataset for machine learning training.

    This function performs the complete data preparation pipeline:
    1. Downloads the UCI Dry Bean dataset
    2. Performs exploratory data analysis with visualizations
    3. Preprocesses the data (encoding categorical variables)
    4. Splits data into train/test sets
    5. Saves data locally and optionally to S3

    Args:
        send_to_s3 (bool): If True, uploads processed data to S3 for SageMaker training.
                          If False, only saves data locally. Defaults to False.

    Returns:
        None

    Note:
        The function creates the following directory structure:
        - data/train/dry-bean-train.csv
        - data/test/dry-bean-test.csv

        If send_to_s3=True, data is also uploaded to S3 with the following structure:
        - s3://bucket/model-name/version/data/train/dry-bean-train.csv
        - s3://bucket/model-name/version/data/test/dry-bean-test.csv
    """
    # Initialize SageMaker session for S3 operations
    sess = sagemaker.Session()

    # Download the UCI Dry Bean dataset
    # This dataset contains 7 different types of dry beans with 16 features
    print("Downloading UCI Dry Bean dataset...")
    dry_bean = fetch_ucirepo(id=602)["data"]["original"]
    print(f"Dataset shape: {dry_bean.shape}")
    print(f"Dataset columns: {list(dry_bean.columns)}")

    # Display first few rows to verify data structure
    dry_bean.head()

    # Create exploratory data analysis visualizations
    print("Creating data visualizations...")

    # Create pairplot for key numerical features to understand relationships
    # Focus on geometric features that are most relevant for bean classification
    sns.pairplot(
        dry_bean,
        vars=[
            "Area",  # Bean area
            "MajorAxisLength",  # Length of major axis
            "MinorAxisLength",  # Length of minor axis
            "Eccentricity",  # Eccentricity of the bean
            "Roundness",  # Roundness of the bean
        ],
        hue="Class",  # Color by bean class for pattern identification
    )

    # Create correlation heatmap to understand feature relationships
    correlation = dry_bean.corr(numeric_only=True)

    # Create a square heatmap with center at 0 for better visualization
    # Use coolwarm colormap to distinguish positive/negative correlations
    sns.heatmap(correlation, center=0, square=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.show()

    # Data preprocessing for machine learning
    print("Preprocessing data for machine learning...")
    df = dry_bean.copy(deep=True)  # Create a copy to avoid modifying original data

    # Encode the target variable (Class) from categorical to numerical
    # This is required for scikit-learn algorithms
    le = LabelEncoder()
    df["Class"] = le.fit_transform(df["Class"])
    print(f"Encoded classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Split the data into training and testing sets
    # Use 80% for training, 20% for testing with a fixed random state for reproducibility
    train, test = train_test_split(df, random_state=1, test_size=0.2)
    print(f"Training set size: {len(train)}")
    print(f"Testing set size: {len(test)}")

    # Create local data directories if they don't exist
    # This ensures the directory structure is available for data storage
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/train").mkdir(parents=True, exist_ok=True)
    Path("data/test").mkdir(parents=True, exist_ok=True)

    # Save processed data locally
    print("Saving data locally...")
    train.to_csv("data/train/dry-bean-train.csv", index=False)
    test.to_csv("data/test/dry-bean-test.csv", index=False)
    print("Data saved to data/train/dry-bean-train.csv and data/test/dry-bean-test.csv")

    if send_to_s3:
        # Upload data to S3 for SageMaker training
        # SageMaker training jobs require data to be in S3
        print("Uploading data to S3...")

        # Upload training data to S3
        trainpath = sess.upload_data(
            path="data/train/dry-bean-train.csv",
            bucket=config.S3_BUCKET_NAME,
            key_prefix=f"{data_paths['data_prefix']}/train",
        )

        # Upload test data to S3
        testpath = sess.upload_data(
            path="data/test/dry-bean-test.csv",
            bucket=config.S3_BUCKET_NAME,
            key_prefix=f"{data_paths['data_prefix']}/test",
        )

        print(f"Training data uploaded to: {trainpath}")
        print(f"Test data uploaded to: {testpath}")
    else:
        print("S3 upload skipped. Data is available locally only.")


if __name__ == "__main__":
    # When run as a script, prepare data and upload to S3
    # This is useful for initial data setup or CI/CD pipelines
    prepare_data(send_to_s3=True)
