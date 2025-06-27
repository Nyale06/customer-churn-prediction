import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("/home/nyale/Documents/personal/customer_churn_prediction/models/churn_model.pkl")
feature_columns = joblib.load("/home/nyale/Documents/personal/customer_churn_prediction/models/feature_columns.pkl")  # ← feature names used during training

st.title("📱 Customer Churn Predictor")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72)
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0)
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# Build input dictionary (dummy variables)
input_dict = {
    "SeniorCitizen": [senior],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    f"gender_{gender}": [1],
    f"InternetService_{internet}": [1],
    f"Contract_{contract}": [1],
    f"PaymentMethod_{payment}": [1],
}

# Convert to DataFrame
input_df = pd.DataFrame(input_dict)

# Ensure all expected features exist
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing features with 0

# Reorder columns to match training data
input_df = input_df[feature_columns]

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    if prediction[0] == 1:
        st.error(f"⚠️ High risk of churn! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer likely to stay. (Probability: {1 - probability:.2%})")
