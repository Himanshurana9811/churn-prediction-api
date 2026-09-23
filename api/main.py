from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap

app = FastAPI(title="Churn Prediction API")

# Load everything once, when the server starts (not on every request - that would be slow)
model = joblib.load("models/churn_model.pkl")
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
numeric_cols = joblib.load("models/numeric_cols.pkl")
categorical_cols = joblib.load("models/categorical_cols.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def engineer_features(data: CustomerData):
    charges_per_tenure = data.TotalCharges / (data.tenure + 1)
    is_new_customer = 1 if data.tenure <= 3 else 0

    service_cols = [data.PhoneService, data.MultipleLines, data.InternetService,
                     data.OnlineSecurity, data.OnlineBackup, data.DeviceProtection,
                     data.TechSupport, data.StreamingTV, data.StreamingMovies]
    total_services = sum(1 for val in service_cols if val not in ["No", "No internet service", "No phone service"])

    contract_risk_flag = 1 if data.Contract == "Month-to-month" else 0

    return charges_per_tenure, is_new_customer, total_services, contract_risk_flag


@app.post("/predict")
def predict_churn(data: CustomerData):
    # Step 1: engineer the 4 custom features
    charges_per_tenure, is_new_customer, total_services, contract_risk_flag = engineer_features(data)

    # Step 2: build a dictionary matching your original raw columns + new features
    input_dict = {
        "SeniorCitizen": data.SeniorCitizen,
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges,
        "charges_per_tenure": charges_per_tenure,
        "is_new_customer": is_new_customer,
        "total_services": total_services,
        "contract_risk_flag": contract_risk_flag
    }

    # Step 3: build the numeric array in the EXACT order numeric_cols expects
    numeric_values = np.array([[input_dict[col] for col in numeric_cols]])

    # Step 4: build a small dataframe for the categorical columns, then encode
    import pandas as pd
    categorical_input = pd.DataFrame([{
        "gender": data.gender, "Partner": data.Partner, "Dependents": data.Dependents,
        "PhoneService": data.PhoneService, "MultipleLines": data.MultipleLines,
        "InternetService": data.InternetService, "OnlineSecurity": data.OnlineSecurity,
        "OnlineBackup": data.OnlineBackup, "DeviceProtection": data.DeviceProtection,
        "TechSupport": data.TechSupport, "StreamingTV": data.StreamingTV,
        "StreamingMovies": data.StreamingMovies, "Contract": data.Contract,
        "PaperlessBilling": data.PaperlessBilling, "PaymentMethod": data.PaymentMethod
    }])
    encoded_values = encoder.transform(categorical_input[categorical_cols])

    # Step 5: scale the numeric values (using the SAME scaler from training)
    scaled_numeric = scaler.transform(numeric_values)

    # Step 6: combine into final input, exactly like training
    final_input = np.hstack([scaled_numeric, encoded_values])

        # Compute SHAP values for this specific customer
    shap_values_single = explainer.shap_values(final_input)[0]
    shap_df = pd.DataFrame({
        "feature": numeric_cols + list(encoder.get_feature_names_out(categorical_cols)),
        "shap_value": shap_values_single
    })
    shap_df["abs_impact"] = shap_df["shap_value"].abs()
    top_reasons = shap_df.sort_values("abs_impact", ascending=False)["feature"].head(3).tolist()


    # Step 7: predict
    probability = model.predict_proba(final_input)[0][1]
    prediction = "High Risk" if probability >= 0.5 else "Low Risk"

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "top_reasons": top_reasons
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running. Visit /docs for interactive documentation."}