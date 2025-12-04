from typing import List
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status
from schemas import PredictionRequest, Explanation


app = FastAPI(
    title="Fraud Detection API",
    description="An API for detecting fraudulent transactions.",
    version="0.1.0"
)

pipeline = joblib.load("models/lr_model.pkl")
explainer = joblib.load("models/lr_explainer.pkl")


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool


@app.get("/")
def get_root():
    return {"status": "ok","message": "Fraud Detection API running"}


@app.get("/health")
def get_health():
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable"
        )
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: PredictionRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Cannot predict"
        )
    
    data = pd.DataFrame([transaction.model_dump()])
    
    fraud_probability = pipeline.predict_proba(data)[:,1]
    is_fraud = bool(fraud_probability >= 0.6)

    return PredictionResponse(
        fraud_probability = fraud_probability, 
        is_fraud = is_fraud)


@app.post("/explain", response_model=List[Explanation])
def explanation_transaction(transaction: PredictionRequest):
    if pipeline is None:
        raise HTTPException(
            status_code= status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = "Model or Explainer not loaded. Service unavailable"
        )
    
    data = pd.DataFrame([transaction.model_dump()])

    preprocessor = pipeline.named_steps['preprocessor']
    scaled_data = preprocessor.transform(data)

    shap_values = explainer.shap_values(scaled_data)
    feature_names = data.columns.tolist()
    values = shap_values[0]
    explanations = [
        Explanation(feature=name, impact=float(val))
        for name, val in zip(feature_names, values)
    ]

    explanations.sort(key=lambda x:abs(x.impact), reverse = True)

    return explanations
