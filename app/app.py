import streamlit as st
import requests
import pandas as pd

# Define the FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/predict"   # Update this to your actual FastAPI endpoint

# Streamlit app title
st.title("Heart Stroke Prediction")

# Create form for input
with st.form("patient_form"):
    st.header("Enter Patient Information")
    id = st.number_input("ID", min_value=0, step=1) 
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    hypertension = st.selectbox("Hypertension", options=[0, 1])
    heart_disease = st.selectbox("Heart Disease", options=[0, 1])
    ever_married = st.selectbox("Ever Married", options=["Yes", "No"])
    work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", options=["formerly smoked", "never smoked", "smokes", "Unknown"])

    # Form submission button
    submit = st.form_submit_button("Predict")

# Process form submission
if submit:
    # Create a dictionary from the form data
    data = {
        "id": id,
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    # Send the data to the FastAPI endpoint and get the prediction
    response = requests.post(FASTAPI_URL, json=data)
    
    if response.status_code == 200:
        result = response.json()
        prediction = result.get("stroke")
        probability = result.get("probability")
        if prediction == 1:
            st.error(f"Potential Heart Stroke (Probability: {probability:.2f})", icon="🚨")
        else:
            st.success(f"Safe (Probability: {probability:.2f})", icon="✅")
    else:
        st.error("Error in prediction", icon="❌")
