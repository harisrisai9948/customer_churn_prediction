import joblib
import pandas as pd
def load_model():

    model = joblib.load(
        "models/logistic_pipeline.pkl"
    )

    return model
def main():

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5,
        "TotalCharges": 1074.0
    }
    prediction, probability = predict_customer(
    sample_customer
    )

    print(f"Prediction: {prediction}")

    print(
    f"Churn Probability: {probability[1]:.2%}"
)

  

def predict_customer(customer_data):

    model = load_model()

    
    customer_df = pd.DataFrame([customer_data])

           

    prediction = model.predict(customer_df)[0]

    probability = model.predict_proba(customer_df)[0]

    return prediction, probability

if __name__ == "__main__":
    main()

