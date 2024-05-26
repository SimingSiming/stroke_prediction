import boto3
import os
import pickle
import logging
import pandas as pd
from io import StringIO

logger = logging.getLogger("heart_stroke")

def save_and_upload_model(model, output_path, bucket_name, s3_path, model_filename):
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
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3_client = session.client('s3')
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    body = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(body))
    return df


def save_data_to_s3(df, bucket_name, file_key, region_name, profile_name):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3_client = session.client('s3')
    s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())
