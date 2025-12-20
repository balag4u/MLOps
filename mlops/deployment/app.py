import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="nsa9/churn-model", filename="best_churn_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Customer Churn Prediction App")
st.write("The Customer Churn Prediction App is an internal tool for bank staff that predicts whether customers are at risk of churning based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to churn.")

# Collect user input
CreditScore = st.number_input("Credit Score (customer's credit score)", min_value=300, max_value=900, value=650)
Geography = st.selectbox("Geography (country where the customer resides)", ["France", "Germany", "Spain"])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (number of years the customer has been with the bank)", value=12)
Balance = st.number_input("Account Balance (customer’s account balance)", min_value=0.0, value=10000.0)
NumOfProducts = st.number_input("Number of Products (number of products the customer has with the bank)", min_value=1, value=1)
HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"])
IsActiveMember = st.selectbox("Is Active Member?", ["Yes", "No"])
EstimatedSalary = st.number_input("Estimated Salary (customer’s estimated salary)", min_value=0.0, value=50000.0)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': 1 if HasCrCard == "Yes" else 0,
    'IsActiveMember': 1 if IsActiveMember == "Yes" else 0,
    'EstimatedSalary': EstimatedSalary
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
