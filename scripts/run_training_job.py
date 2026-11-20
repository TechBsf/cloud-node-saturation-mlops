# scripts/run_training_job.py
# Same estimator as create_pipeline.py, but launched as a plain training job
# (no SageMaker Pipelines).

import os
import time
import boto3
import sagemaker

from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

REGION = os.environ["AWS_REGION"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["S3_BUCKET"]
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")

# Use our own bucket so the SDK does not try to create a default one
sess = sagemaker.Session(
    boto_session=boto3.Session(region_name=REGION),
    default_bucket=BUCKET,
)

train_s3_uri = f"s3://{BUCKET}/data/compute_capacity.csv"
env = {"MLFLOW_TRACKING_URI": MLFLOW_URI} if MLFLOW_URI else {}

estimator = SKLearn(
    entry_point="train_with_mlflow.py",
    source_dir="scripts",
    role=ROLE_ARN,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sess,
    environment=env,
    output_path=f"s3://{BUCKET}/models/",
    code_location=f"s3://{BUCKET}/code",
)

job_name = f"compute-capacity-sklearn-{int(time.time())}"
print("Starting training job:", job_name)
estimator.fit(
    {"train": TrainingInput(s3_data=train_s3_uri, content_type="text/csv")},
    job_name=job_name,
    wait=True,
)
print("Model artifacts:", estimator.model_data)

