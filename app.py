from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open('config/params.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
prediction_errors = Counter('prediction_errors_total', 'Total prediction errors')

# Initialize FastAPI app
app = FastAPI(
    title="Stock Return Prediction API",
    description="MLOps API for predicting 3-year stock returns",
    version="1.0.0"
)

# Global variables for model and artifacts
model = None
scaler = None
label_encoders = None
feature_columns = None


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, float] = Field(..., description="Feature values as key-value pairs")
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "feature1": 100.5,
                    "feature2": 0.25,
                    "feature3": 1500
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    model_version: str
    timestamp: str
    confidence_interval: Dict[str, float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str


def load_artifacts():
    """Load model and preprocessing artifacts"""
    global model, scaler, label_encoders, feature_columns
    
    try:
        model_dir = Path("models/")
        
        # Load model
        model = joblib.load(model_dir / "best_random_forest_model.pkl")
        logger.info("Model loaded successfully")
        
        # Load scaler
        scaler = joblib.load(model_dir / "scaler.pkl")
        logger.info("Scaler loaded successfully")
        
        # Load label encoders
        with open(model_dir / "label_encoders.pkl", 'rb') as f:
            label_encoders = joblib.load(f)
        logger.info("Label encoders loaded successfully")
        
        # Load feature columns
        with open(model_dir / "feature_columns.pkl", 'rb') as f:
            feature_columns = joblib.load(f)
        logger.info("Feature columns loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up API...")
    success = load_artifacts()
    if not success:
        logger.error("Failed to load model artifacts")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Return Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Convert request to DataFrame
        input_df = pd.DataFrame([request.features])
        
        # Ensure correct feature order
        if not all(col in input_df.columns for col in feature_columns):
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features. Expected: {feature_columns}"
            )
        
        input_df = input_df[feature_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Calculate confidence interval (using std of predictions from trees)
        if hasattr(model, 'estimators_'):
            tree_predictions = np.array([tree.predict(input_scaled)[0] for tree in model.estimators_])
            ci_lower = np.percentile(tree_predictions, 2.5)
            ci_upper = np.percentile(tree_predictions, 97.5)
            confidence_interval = {
                "lower": float(ci_lower),
                "upper": float(ci_upper)
            }
        else:
            confidence_interval = None
        
        # Update metrics
        prediction_counter.inc()
        prediction_duration.observe(time.time() - start_time)
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version="1.0.0",
            timestamp=datetime.now().isoformat(),
            confidence_interval=confidence_interval
        )
        
    except Exception as e:
        prediction_errors.inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        for request in requests:
            result = await predict(request)
            predictions.append(result)
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/reload")
async def reload_model():
    """Reload model artifacts"""
    success = load_artifacts()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        log_level="info"
    )
