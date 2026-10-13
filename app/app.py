import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# Find the saved model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "customer_churn_model.pkl"
)

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction")
st.write(
    "Enter the customer details below to predict "
    "whether the customer is likely to churn."
)

# Customer information
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender", ["Female", "Male"]
    )
    senior_citizen = st.selectbox(
        "Senior Citizen", [0, 1]
    )
    partner = st.selectbox(
        "Has Partner?", ["Yes", "No"]
    )
    dependents = st.selectbox(
        "Has Dependents?", ["Yes", "No"]
    )
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )
    phone_service = st.selectbox(
        "Phone Service", ["Yes", "No"]
    )
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

# Build input in the same feature format used during training
customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

if st.button("Predict Churn", type="primary"):
    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.warning(
            "The model predicts that this customer "
            "is likely to churn."
        )
    else:
        st.success(
            "The model predicts that this customer "
            "is unlikely to churn."
        )

    st.caption(
        "This is a machine learning prediction, "
        "not a guarantee of future customer behavior."
    )