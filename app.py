import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the saved model and scaler
model = tf.keras.models.load_model('churn_model.h5')
sc = joblib.load('scaler.joblib')

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Enter customer details below to see if they are likely to leave the bank.")

# Input Grid
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    credit_score = st.slider("Credit Score", 300, 850, 600)
    age = st.number_input("Age", 18, 95, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)

with col2:
    balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_card = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active = st.radio("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0)

# Prediction Logic
if st.button("Calculate Churn Risk"):
    # Encoding to match your training logic
    # [France, Germany, Spain]
    geo_map = {"France": [1, 0, 0], "Germany": [0, 1, 0], "Spain": [0, 0, 1]}
    gen_val = 1 if gender == "Male" else 0
    card_val = 1 if has_card == "Yes" else 0
    active_val = 1 if is_active == "Yes" else 0
    
    # Combine into a single row
    features = geo_map[geography] + [credit_score, gen_val, age, tenure, balance, num_products, card_val, active_val, salary]
    
    # Scale and Predict
    processed_features = sc.transform([features])
    prediction = model.predict(processed_features)
    churn_chance = prediction[0][0]

    st.divider()
    if churn_chance > 0.5:
        st.error(f"⚠️ **High Risk:** This customer is likely to leave. (Probability: {churn_chance:.2%})")
    else:
        st.success(f"✅ **Low Risk:** This customer is likely to stay. (Probability: {1-churn_chance:.2%})")