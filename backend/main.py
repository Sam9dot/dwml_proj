"""
FastAPI Backend for Healthcare Cost Clustering & Anomaly Detection
Loads artifacts from the single models/model.joblib file
"""

import os
import joblib
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Healthcare Cost Clustering & Anomaly Detection API",
    description="K-Means (Customer Segmentation) & DBSCAN (Atypical Billing Detection)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load single artifact file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("models", "model.joblib")

try:
    artifacts = joblib.load(MODEL_PATH)
    scaler = artifacts['scaler']
    kmeans = artifacts['kmeans']
    dbscan_core_samples = artifacts['dbscan_core_samples']
    dbscan_eps = artifacts['dbscan_eps']
    cluster_profiles = artifacts['cluster_profiles']
    metrics_table = artifacts['metrics_table']
    print(f"Backend loaded single model artifact: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model artifact: {e}")
    artifacts = None
    scaler = None
    kmeans = None


class PolicyholderInput(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Age in years")
    bmi: float = Field(..., ge=10.0, le=65.0, description="BMI kg/m²")
    smoker: str = Field(..., description="'yes' or 'no'")
    children: Optional[int] = Field(0, ge=0, le=10)
    charges: Optional[float] = Field(None, ge=0.0)


@app.get("/")
def home():
    return {
        "status": "online",
        "project": "Healthcare Cost Clustering & Unusual Billing Detection",
        "algorithms": "K-Means (K=4) & DBSCAN (eps=0.58)",
        "endpoints": ["/predict", "/summary", "/docs"]
    }


@app.get("/summary")
def get_summary():
    """Returns cluster personas and the 3 simple metrics"""
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model artifact not loaded")
    return {
        "metrics": metrics_table,
        "profiles": cluster_profiles,
        "total_records": len(artifacts['df'])
    }


@app.post("/predict")
def predict_cluster_and_anomaly(data: PolicyholderInput):
    """Assigns K-Means cluster and checks DBSCAN core distance for unusual billing"""
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    smoker_enc = 1 if str(data.smoker).strip().lower() in ["yes", "y", "1"] else 0

    # Default charges baseline if not provided
    if data.charges is None or data.charges <= 0:
        if smoker_enc == 1:
            input_charges = 32000.0
        elif data.age >= 45:
            input_charges = 12600.0
        elif data.bmi >= 32:
            input_charges = 6000.0
        else:
            input_charges = 5300.0
    else:
        input_charges = float(data.charges)

    raw = np.array([[data.age, data.bmi, smoker_enc, input_charges]], dtype=float)
    scaled = scaler.transform(raw)

    # 1. K-Means
    cluster_id = int(kmeans.predict(scaled)[0])
    profile = next((p for p in cluster_profiles if p['cluster_id'] == cluster_id), cluster_profiles[cluster_id])

    # 2. DBSCAN Unusual check: distance to nearest core point
    distances = np.linalg.norm(dbscan_core_samples - scaled, axis=1)
    min_dist = float(np.min(distances))
    is_unusual = bool(min_dist > dbscan_eps)

    if is_unusual:
        verdict = "Atypical Record - Flagged for Review (DBSCAN Outlier)"
        explanation = f"Record distance ({min_dist:.2f}) exceeds cluster density boundary (eps={dbscan_eps:.2f})."
    else:
        verdict = "Normal Pattern (Within Cluster Neighborhood)"
        explanation = f"Record matches standard group density (distance {min_dist:.2f} <= eps {dbscan_eps:.2f})."

    return {
        "cluster_id": cluster_id,
        "persona_name": profile["persona_name"],
        "cost_tier": profile["cost_tier"],
        "group_avg_charges": profile["avg_charges"],
        "group_median_charges": profile["median_charges"],
        "group_smoker_pct": profile["smoker_pct"],
        "input_charges": round(input_charges, 2),
        "is_unusual": is_unusual,
        "nearest_core_distance": round(min_dist, 3),
        "eps_threshold": round(dbscan_eps, 2),
        "anomaly_verdict": verdict,
        "anomaly_explanation": explanation
    }
