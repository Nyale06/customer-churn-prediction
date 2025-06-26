import streamlit as st
import pandas as pd
import joblib

model = joblib.load("/home/nyale/Documents/personal/customer_churn_prediction/models/churn_model.pkl")

st.title("📱 Customer Churn Predictor")

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

# Convert input to DataFrame (ensure this matches training structure + preprocessing!)
input_dict = {
    "SeniorCitizen": [senior],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    "gender_Female": [1 if gender == "Female" else 0],
    "gender_Male": [1 if gender == "Male" else 0],
    "InternetService_DSL": [1 if internet == "DSL" else 0],
    "InternetService_Fiber optic": [1 if internet == "Fiber optic" else 0],
    "InternetService_No": [1 if internet == "No" else 0],
    "Contract_Month-to-month": [1 if contract == "Month-to-month" else 0],
    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],
    "PaymentMethod_Electronic check": [1 if payment == "Electronic check" else 0],
    "PaymentMethod_Mailed check": [1 if payment == "Mailed check" else 0],
    "PaymentMethod_Bank transfer": [1 if payment == "Bank transfer" else 0],
    "PaymentMethod_Credit card": [1 if payment == "Credit card" else 0],
}

input_df = pd.DataFrame(input_dict)

if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    if prediction[0] == 1:
        st.error(f"⚠️ High risk of churn! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Customer is likely to stay. (Probability: {1 - probability:.2%})")
