# Customer Churn Prediction & Retention API

An end-to-end, explainable machine learning service that predicts customer churn risk and explains *why* — built and deployed as a production-style API, not just a notebook.

**Live API:** https://churn-prediction-api-8yy3.onrender.com/docs
*(Free-tier hosting — the first request may take 30–60s to wake up if idle.)*

---

## The Problem

Customer churn is expensive to react to and cheap to prevent — if you know who's at risk early enough. This project predicts the probability that a telecom customer will churn, and explains which specific factors are driving that risk for each individual customer, so a retention team could act on it.

## What Makes This Different

Most churn prediction projects stop at "I trained a model in a notebook." This one:
- Compares 4 models with tracked, reproducible experiments (not just one algorithm)
- Explains *individual* predictions with SHAP, not just global feature importance
- Ships as a real, callable API — validated input, live cloud deployment, containerized
- Every engineered feature is validated with evidence across multiple models before being trusted

## Example Request/Response

**Request:**
```json
{
  "gender": "Female",
  "tenure": 2,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "MonthlyCharges": 85.5,
  "TotalCharges": 170.0,
  "...": "(see /docs for full schema)"
}
```

**Response:**
```json
{
  "churn_probability": 0.7082,
  "prediction": "High Risk",
  "top_reasons": ["tenure", "InternetService_Fiber optic", "total_services"]
}
```

## Model Comparison

Four models were trained and evaluated on the same engineered feature set; **Logistic Regression was selected** for production — it generalized best and showed the lowest overfitting, despite being the simplest of the four.

| Metric | Decision Tree | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|---|
| Test accuracy | 0.782 | **0.812** | 0.804 | 0.793 |
| Precision | 0.670 | 0.684 | **0.675** | 0.635 |
| Recall | 0.353 | **0.543** | 0.505 | 0.521 |
| F1 | 0.462 | **0.605** | 0.578 | 0.573 |
| Overfitting gap | 0.009 | -0.003 | 0.015 | 0.037 |

## Feature Engineering

Four custom features were engineered from domain insight and validated with evidence (not assumption) across multiple models:
- `charges_per_tenure` — average spend rate over the customer's lifetime
- `is_new_customer` — flag for customers in their first 3 months
- `total_services` — count of bundled services subscribed
- `contract_risk_flag` — flag for month-to-month contracts (proved to be the single strongest churn signal)

## Explainability

Every prediction is accompanied by its top 3 SHAP-driven reasons, giving a human-readable explanation for that specific customer — not just a black-box probability.

## Tech Stack

- **Modeling:** scikit-learn, XGBoost, pandas, NumPy
- **Explainability:** SHAP
- **Experiment tracking:** MLflow
- **API:** FastAPI, Pydantic, Uvicorn
- **Deployment:** Docker, Render
- **Versioning:** Git/GitHub

## Project Structure

```
churn-prediction-api/
├── api/                # FastAPI service (main.py)
├── data/                # Dataset
├── models/              # Packaged model, encoder, scaler, SHAP explainer
├── notebooks/            # EDA, feature engineering, model comparison
├── Dockerfile
├── requirements-api.txt   # Minimal deps for the API/Docker image
└── requirements.txt      # Full dev environment
```

## Run Locally

```bash
git clone https://github.com/Himanshurana9811/churn-prediction-api.git
cd churn-prediction-api

# Using Docker (recommended - matches production exactly)
docker build -t churn-prediction-api .
docker run -d -p 8000:8000 --name churn-api churn-prediction-api
```
Then visit `http://127.0.0.1:8000/docs`

## Author

Himanshu Rana — [GitHub](https://github.com/Himanshurana9811)
