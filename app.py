import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import os, joblib
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.pkl"))

# Load scaler and feature selector
with open('std.pkl', 'rb') as f:
    scaler = pickle.load(f)

try:
    with open('feature.pkl', 'rb') as f:
        feature_selector = pickle.load(f)
        st.write("Model expects features:", feature_selector)
except:
    feature_selector = None

# ------------------ Streamlit UI ------------------
st.title("💳 Credit Score Prediction")
st.markdown("### Enter the necessary details below:")

with st.sidebar:
    st.header("📊 Input Features")

    Occupation = st.slider("Occupation (encoded)", 1, 15, step=1)
    Num_of_Loan = st.slider("Number of Loans", 1, 450, step=1)
    Delay_from_due_date = st.slider("Delay from Due Date (days)", -1, 70, step=1)
    Credit_Mix = st.selectbox("Credit Mix (0=Bad, 1=Standard, 2=Good, 3=Excellent)", [0, 1, 2, 3])
    Credit_Utilization_Ratio = st.slider("Credit Utilization Ratio (%)", 15.0, 50.0, step=0.1)
    Payment_of_Min_Amount = st.selectbox("Payment of Minimum Amount (0=No, 1=Yes, 2=NM)", [0, 1, 2])
    Payment_Behaviour = st.selectbox("Payment Behaviour", [0, 1, 2, 3, 4, 5, 6])

    submit = st.button("⏳ Predict Credit Score")

# ------------------ Prediction Logic ------------------
if submit:
    try:
        # Combine all features in the correct order
        feature = np.array([[Occupation, Num_of_Loan, Delay_from_due_date, Credit_Mix,
                             Credit_Utilization_Ratio, Payment_of_Min_Amount, Payment_Behaviour]])

        # Scale features
        scaled_feature = scaler.transform(feature)

        # Predict using model
        prediction = model.predict(scaled_feature)

        # Output result with emojis
        if prediction[0] == "Good":
            result = "🟢 GOOD Score"
        elif prediction[0] == "Poor":
            result = "🔴 POOR Score"
        else:
            result = "🟡 STANDARD Score"

        st.success(f"Prediction Result: {result}")

        # Show debug info
        st.markdown("### 🔍 Debug Info")
        st.write("Input Features (raw):", feature)
        st.write("Scaled Features:", scaled_feature)
        st.write("Model Prediction:", prediction)

    except Exception as e: 
        st.error(f"Error: {e}")
