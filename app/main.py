from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import boto3
import pandas as pd
from io import BytesIO
import src.preprocess as pp

app = FastAPI()

# Load model from S3
def load_model_from_s3(bucket_name, model_key, region_name='us-east-2'):
    session = boto3.Session(profile_name='mlds_ce', region_name=region_name)
    s3 = session.client('s3', region_name=region_name)
    obj = s3.get_object(Bucket=bucket_name, Key=model_key)
    model = joblib.load(BytesIO(obj['Body'].read()))
    return model

# Load the models
models = {
    "random_forest_model": load_model_from_s3('cloud-engineer-bucket', 'models/random_forest_model.pkl'),
    "lr_model": load_model_from_s3('cloud-engineer-bucket', 'models/lr_model.pkl'),
    "xgb_model": load_model_from_s3('cloud-engineer-bucket', 'models/xgb_model.pkl'),
}

class PatientInfo(BaseModel):
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
    model_type: str

@app.post("/predict")
def predict(patient: PatientInfo):
    if patient.model_type not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = models[patient.model_type]

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
    numeric_features = ['age', 'avg_glucose_level','bmi']
    cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    drop_features =['id','heart_disease']

    # Apply the preprocessing function
    X_transformed_df = pp.preprocess_data(df, numeric_features, cat_features, drop_features)

    # Make prediction
    prediction = model.predict(X_transformed_df)
    probability = model.predict_proba(X_transformed_df)[:, 1]  # Assuming binary classification and we're interested in the probability of the positive class (stroke)

    return {"stroke": int(prediction[0]), "probability": float(probability[0])}
