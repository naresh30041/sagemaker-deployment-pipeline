# deploy.py
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import boto3

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE"  # Replace with your SageMaker IAM role

# S3 path to the model artifact (set by the training job)
model_data = "s3://your-bucket-name/model/model.tar.gz"  # Replace with your actual S3 path

# Create the SKLearnModel object
sklearn_model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point="train.py",
    framework_version="0.23-1",
    sagemaker_session=sagemaker_session
)

# Deploy the model to an endpoint
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

print(f"Model deployed to endpoint {predictor.endpoint_name}")