"""
This module provides a FastAPI app for predicting stroke risk based on patient information.

It contains endpoints for receiving patient data, preprocessing it, making predictions, and handling exceptions.

Endpoints:
    - POST /predict: Predicts the likelihood of a stroke given patient information.

Exception Handlers:
    - http_exception_handler: Handles HTTP exceptions.
    - generic_exception_handler: Handles generic exceptions.
"""
import logging
import logging.config
from io import BytesIO
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import joblib
import boto3
import pandas as pd  # Added missing import
import src.preprocess as pp

# Setup logging
def setup_logging(config_path='pipeline/config/config.yaml', default_level=logging.INFO):
    """Setup logging configuration."""
    try:
        with open(config_path, 'rt', encoding='utf-8') as file:
            config = yaml.safe_load(file.read())
        logging.config.dictConfig(config['logging'])
    except Exception as log_error:
        print(f"Error in logging configuration: {log_error}")
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
        s3_client = session.client('s3', region_name=region_name)
        obj = s3_client.get_object(Bucket=bucket_name, Key=model_key)
        model = joblib.load(BytesIO(obj['Body'].read()))
        return model
    except Exception as load_error:
        logger.error("Error loading model from S3", exc_info=True)
        raise HTTPException(status_code=500, detail="Model could not be loaded") from load_error

# Load the model
try:
    model = load_model_from_s3('cloud-engineer-bucket', 'models/random_forest_model.pkl')
    logger.info("Model loaded successfully")
except Exception as model_load_error:
    logger.error("Failed to load model", exc_info=True)

class PatientInfo(BaseModel):
    """Model for patient information."""
    patient_id: int  # Changed from 'id' to 'patient_id'
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str  # Changed from 'Residence_type' to 'residence_type'
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
            'id': [patient.patient_id],
            'gender': [patient.gender],
            'age': [patient.age],
            'hypertension': [patient.hypertension],
            'heart_disease': [patient.heart_disease],
            'ever_married': [patient.ever_married],
            'work_type': [patient.work_type],
            'residence_type': [patient.residence_type],
            'avg_glucose_level': [patient.avg_glucose_level],
            'bmi': [patient.bmi],
            'smoking_status': [patient.smoking_status]
        }
        df = pd.DataFrame(data)

        # Apply the preprocessing function
        numeric_features = ['age', 'avg_glucose_level', 'bmi']
        cat_features = ['gender', 'ever_married', 'work_type', 'residence_type', 'smoking_status']
        drop_features = ['id', 'heart_disease']
        x_transformed_df = pp.preprocess_data(df, numeric_features, cat_features, drop_features)

        # Make prediction
        prediction = model.predict(x_transformed_df)
        probability = model.predict_proba(x_transformed_df)[:, 1]  # Assuming binary classification

        logger.info("Prediction made successfully")
        return {"stroke": int(prediction[0]), "probability": float(probability[0])}
    except ValidationError as validation_error:
        logger.error("Validation error", exc_info=True)
        raise HTTPException(status_code=422, detail=validation_error.errors()) from validation_error
    except Exception as prediction_error:
        logger.error("Error during prediction", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction could not be made") from prediction_error

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error("HTTP error occurred: %s", exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions."""
    logger.error("Unexpected error: %s", str(exc), exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An unexpected error occurred"})