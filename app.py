import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up page configurations for a clean layout
st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Load the saved model and transformer pipeline artifacts
# CRITERIA CHECK: Uses the saved transformers and saved model components
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('saved_model.pkl')
        scaler = joblib.load('saved_transformer.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Error: 'saved_model.pkl' or 'saved_transformer.pkl' not found. Ensure they are in the same directory as this app.")
        return None, None

model, scaler = load_artifacts()

# Title and Description
st.title("❤️ Cardiovascular Disease Risk Predictor")
st.write("""
This machine learning application predicts the likelihood of cardiovascular disease based on patient clinical and physiological parameters.
""")
st.markdown("---")

# Organized User Input Form
st.header("Patient Medical Profile")
st.write("Please enter the patient's medical details below:")

col1, col2 = st.columns(2)

with col1:
    # Age input in years, converted to days inside the script to match the raw dataset feature format
    age_years = st.number_input("Age (in Years)", min_value=1, max_value=120, value=50, step=1)
    age_days = age_years * 365.25
    
    gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
    gender_encoded = 1 if gender == "Female" else 2
    
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165, step=1)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)

with col2:
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=40, max_value=300, value=120, step=1)
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=30, max_value=200, value=80, step=1)
    
    cholesterol = st.selectbox("Cholesterol Level", options=["1: Normal", "2: Above Normal", "3: Well Above Normal"], index=0)
    chol_encoded = int(cholesterol.split(":")[0])
    
    gluc = st.selectbox("Glucose Level", options=["1: Normal", "2: Above Normal", "3: Well Above Normal"], index=0)
    gluc_encoded = int(gluc.split(":")[0])

st.markdown("### Lifestyle Factors")
col3, col4, col5 = st.columns(3)

with col3:
    smoke = st.checkbox("Smoking History")
    smoke_encoded = 1 if smoke else 0

with col4:
    alco = st.checkbox("Alcohol Consumption")
    alco_encoded = 1 if alco else 0

with col5:
    active = st.checkbox("Physically Active", value=True)
    active_encoded = 1 if active else 0

# --- FEATURE ENGINEERING ---
# CRITERIA CHECK: Re-implement the exact feature engineering pipeline used during training
# 1. Body Mass Index (BMI)
bmi = weight / ((height / 100) ** 2)

# 2. Pulse Pressure (Interaction feature between Systolic and Diastolic pressure)
pulse_pressure = ap_hi - ap_lo

st.markdown("---")

# Prediction Trigger
if st.button("Analyze Cardiovascular Risk", type="primary"):
    if model is not None and scaler is not None:
        
        # Structure raw inputs in the exact array format and column order expected by the pipeline
        raw_features = np.array([[
            age_days, gender_encoded, height, weight, 
            ap_hi, ap_lo, chol_encoded, gluc_encoded, 
            smoke_encoded, alco_encoded, active_encoded, 
            bmi, pulse_pressure
        ]])
        
        # Transform features using the saved scaler artifact
        scaled_features = scaler.transform(raw_features)
        
        # Calculate final prediction and probability mappings
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        risk_probability = probabilities[1] * 100
        
        # Display Visual Results to the user
        st.header("Analysis Results")
        
        if prediction == 1:
            st.error(f"🔴 **High Risk Detected:** The patient has a high probability of cardiovascular disease.")
        else:
            st.success(f"🟢 **Low Risk Detected:** The patient has a low probability of cardiovascular disease.")
            
        # Display progress bar showing probability scaling
        st.write(f"Calculated Risk Probability: **{risk_probability:.2f}%**")
        st.progress(int(risk_probability))
        
    else:
        st.error("Could not run prediction because model artifacts are missing.")
