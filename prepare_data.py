import boto3
import pandas as pd
import seaborn as sns
import sagemaker
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import Config

# Get configuration
config = Config()

# Get the model version from version.txt
model_version = config.get_model_version()

# Get data paths
data_paths = config.get_data_paths()

sm_boto3 = boto3.client("sagemaker")
sess = sagemaker.Session()

region = sess.boto_session.region_name

dry_bean = pd.read_csv(data_paths["data_path"] + "/raw/dry_bean.csv")
dry_bean.head()

sns.pairplot(
    dry_bean,
    vars=["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "roundness"],
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

train.to_csv("dry-bean-train.csv")
test.to_csv("dry-bean-test.csv")

# Send data to S3. SageMaker will take training data from s3
trainpath = sess.upload_data(
    path="dry-bean-train.csv",
    bucket=config.S3_BUCKET_NAME,
    key_prefix=f"{data_paths['data_prefix']}/train",
)

testpath = sess.upload_data(
    path="dry-bean-test.csv",
    bucket=config.S3_BUCKET_NAME,
    key_prefix=f"{data_paths['data_prefix']}/test",
)
