import boto3
import logging
import pandas as pd
from io import StringIO

logger = logging.getLogger("heart_stroke")

def upload_file_to_s3(local_path, bucket, s3_path, region_name, profile_name='default'):
    try:
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        s3_client = session.client('s3')
        s3_client.upload_file(local_path, bucket, s3_path)
        logger.info(f"File successfully uploaded to S3 at {s3_path}")
    except Exception as e:
        logger.error(f"Error uploading file to S3: {e}")
        raise
    
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
