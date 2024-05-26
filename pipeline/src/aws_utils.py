"""
This module provides utility functions for working with AWS services, including S3 storage.

"""

import boto3
import os
import pickle
import logging
import pandas as pd
from io import StringIO, BytesIO

logger = logging.getLogger("heart_stroke")

def save_and_upload_model(model, output_path, bucket_name, s3_path, model_filename):
    """
    Save a machine learning model to a local file and upload it to an S3 bucket.

    Args:
        model: The machine learning model object to be saved.
        output_path (str): The local directory path to save the model file.
        bucket_name (str): The name of the S3 bucket to upload the model to.
        s3_path (str): The S3 key (path) to save the model file.
        model_filename (str): The filename to use for the saved model file.

    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save the model to a .pkl file
    model_file = os.path.join(output_path, model_filename)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    # Upload the model to S3
    s3 = boto3.client('s3')
    s3.upload_file(model_file, bucket_name, os.path.join(s3_path, model_filename))
    print(f"Model uploaded to s3://{bucket_name}/{os.path.join(s3_path, model_filename)}")
    
def load_data_from_s3(bucket_name, file_key, region_name, profile_name):
    """
    Load data from a CSV file stored in an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        file_key (str): The key (path) to the CSV file in the bucket.
        region_name (str): The AWS region where the bucket is located.
        profile_name (str): The AWS profile name to use for authentication.

    Returns:
        DataFrame: The loaded data as a Pandas DataFrame.

    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3_client = session.client('s3')
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    body = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(body))
    return df

def save_data_to_s3(df, bucket_name, file_key, region_name, profile_name):
    """
    Save a DataFrame to a CSV file and upload it to an S3 bucket.

    Args:
        df (DataFrame): The DataFrame to be saved.
        bucket_name (str): The name of the S3 bucket.
        file_key (str): The key (path) to save the CSV file in the bucket.
        region_name (str): The AWS region where the bucket is located.
        profile_name (str): The AWS profile name to use for authentication.

    """
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3_client = session.client('s3')
    s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())

def load_model_from_s3(bucket_name, model_key, region_name='us-east-2', profile_name=None):
    """
    Load a machine learning model from an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        model_key (str): The key (path) to the model file in the bucket.
        region_name (str): The AWS region where the bucket is located.
        profile_name (str): The AWS profile name to use for authentication.

    Returns:
        model: The loaded machine learning model.

    """
    print(f"Loading model from S3 bucket: {bucket_name}, key: {model_key}")
    
    if profile_name:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
    else:
        session = boto3.Session(region_name=region_name)

    s3 = session.client('s3')
    
    try:
        with open('temp_model.pkl', 'wb') as data:
            s3.download_fileobj(bucket_name, model_key, data)
        with open('temp_model.pkl', 'rb') as data:
            model = pickle.load(data)
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        raise e

def upload_artifacts_to_s3(directory, bucket_name, s3_prefix, region_name, profile_name):
    """
    Upload all files in a directory to an S3 bucket.

    Args:
        directory (str): Path to the local directory containing the artifacts.
        bucket_name (str): S3 bucket name.
        s3_prefix (str): S3 prefix (directory) to upload the artifacts to.
        region_name (str): AWS region name.
        profile_name (str): AWS profile name.

    """
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3 = session.client('s3')
    for root, dirs, files in os.walk(directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, directory)
            s3_path = os.path.join(s3_prefix, relative_path)
            s3.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")
