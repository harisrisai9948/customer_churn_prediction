import streamlit as st
import joblib
import pandas as pd

from pathlib import Path

MODEL_PATH = (
    Path(__file__).parent.parent
    / "models"
    / "logistic_pipeline.pkl"
)

model = joblib.load(MODEL_PATH)

st.title(
    "Customer Churn Predictor"
)

st.write(
    "Predict whether a customer is likely to churn."
)
st.header("Customer Information")
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=89.5
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1074.0
)
partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)
st.header("Services")
phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
st.header("Billing")

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
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
senior_citizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)
if st.button("Predict Churn"):
    customer = {
        
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
    }
    customer_df = pd.DataFrame(
             [customer]
    )

    prediction = model.predict(
        customer_df
    )[0]

    probability = model.predict_proba(
        customer_df
    )[0][1]

    st.subheader(
        "Prediction"
    )

    st.write(
        f"Churn Prediction: {prediction}"
    )

    st.write(
        f"Churn Probability: {probability:.2%}"
    )
    if prediction == "Yes":
         st.error(
             f"High Churn Risk ({probability:.2%})"
    )
    else:
        st.success(
            f"Low Churn Risk ({probability:.2%})"
    )