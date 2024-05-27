"""
This module serves as the entry point for the FastAPI application.

It loads a machine learning model from an S3 bucket, defines a FastAPI app with an endpoint for making predictions
based on patient information, and starts the server using Uvicorn.

"""

import logging
import logging.config
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import boto3
import pandas as pd
from io import BytesIO
import src.preprocess as pp

def setup_logging(config_path='pipeline/config/config.yaml', default_level=logging.INFO):
    """
    Setup logging configuration.

    Args:
        config_path (str): The path to the logging configuration file.
        default_level (int): The default logging level.

    """
    try:
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config['logging'])
    except Exception as e:
        print(f"Error in logging configuration: {e}")
        logging.basicConfig(level=default_level)

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

def load_model_from_s3(bucket_name, model_key, region_name='us-east-2'):
    """
    Load a model from an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        model_key (str): The key (path) to the model file in the bucket.
        region_name (str): The AWS region where the bucket is located.

    Returns:
        model: The loaded model.

    """
    try:
        session = boto3.Session(profile_name='mlds_ce', region_name=region_name)
        s3 = session.client('s3', region_name=region_name)
        obj = s3.get_object(Bucket=bucket_name, Key=model_key)
        model = joblib.load(BytesIO(obj['Body'].read()))
        return model
    except Exception as e:
        logger.error("Error loading model from S3", exc_info=True)
        raise HTTPException(status_code=500, detail="Model could not be loaded")

# Load the model
try:
    model = load_model_from_s3('cloud-engineer-bucket', 'models/random_forest_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Failed to load model", exc_info=True)

class PatientInfo(BaseModel):
    """
    Model for patient information.
    """
    id: int
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.post("/predict")
def predict(patient: PatientInfo):
    """
    Predict the likelihood of a stroke given patient information.

    Args:
        patient (PatientInfo): The patient information.

    Returns:
        dict: The prediction result containing the stroke prediction and probability.

    """
    try:
        # Create a DataFrame from the input data
        data = {
            'id': [patient.id],
            'gender': [patient.gender],
            'age': [patient.age],
            'hypertension': [patient.hypertension],
            'heart_disease': [patient.heart_disease],
            'ever_married': [patient.ever_married],
            'work_type': [patient.work_type],
            'Residence_type': [patient.Residence_type],
            'avg_glucose_level': [patient.avg_glucose_level],
            'bmi': [patient.bmi],
            'smoking_status': [patient.smoking_status]
        }
        df = pd.DataFrame(data)

        # Apply the preprocessing function
        numeric_features = ['age', 'avg_glucose_level', 'bmi']
        cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        drop_features = ['id', 'heart_disease']
        X_transformed_df = pp.preprocess_data(df, numeric_features, cat_features, drop_features)

        # Make prediction
        prediction = model.predict(X_transformed_df)
        probability = model.predict_proba(X_transformed_df)[:, 1]  # Assuming binary classification

        logger.info("Prediction made successfully")
        return {"stroke": int(prediction[0]), "probability": float(probability[0])}
    except Exception as e:
        logger.error("Error during prediction", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction could not be made")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
