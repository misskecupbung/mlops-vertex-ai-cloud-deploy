"""
ML Model Serving API
This script serves the trained Iris classification model via REST API.
"""

import json
import logging
import os
import pickle
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import storage
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="ML model serving API for Iris flower classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
labels = None
model_version = os.getenv('MODEL_VERSION', 'v1')


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[float] = Field(
        ...,
        description="List of 4 features: sepal_length, sepal_width, petal_length, petal_width",
        min_items=4,
        max_items=4,
        example=[5.1, 3.5, 1.4, 0.2]
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    instances: List[List[float]] = Field(
        ...,
        description="List of feature vectors for batch prediction",
        example=[[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: str
    confidence: float
    probabilities: dict
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[dict]
    model_version: str


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    model_version: str


def download_model_from_gcs(bucket_name: str, prefix: str, local_dir: str = '/tmp/model'):
    """Download model artifacts from GCS."""
    logger.info(f"Downloading model from gs://{bucket_name}/{prefix}/")
    
    os.makedirs(local_dir, exist_ok=True)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        filename = os.path.basename(blob.name)
        if filename:
            local_path = os.path.join(local_dir, filename)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded {filename}")
    
    return local_dir


def load_model(model_dir: str):
    """Load model and labels from local directory."""
    global model, labels
    
    # Load model
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
    
    # Load labels
    labels_path = os.path.join(model_dir, 'labels.json')
    with open(labels_path, 'r') as f:
        labels = json.load(f)['labels']
    logger.info(f"Labels loaded: {labels}")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model, labels
    
    # Get model location from environment
    model_uri = os.getenv('MODEL_URI', '')
    
    if model_uri.startswith('gs://'):
        # Download from GCS
        parts = model_uri.replace('gs://', '').split('/', 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''
        model_dir = download_model_from_gcs(bucket_name, prefix)
        load_model(model_dir)
    elif os.path.exists('/model'):
        # Load from local volume mount
        load_model('/model')
    else:
        logger.warning("No model found. Server will start without model.")
        logger.warning("Set MODEL_URI environment variable to load model from GCS")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "Iris Classification API",
        "version": model_version,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction_idx = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get class label
        prediction_label = labels[prediction_idx]
        confidence = float(probabilities[prediction_idx])
        
        # Create probability dict
        prob_dict = {label: float(prob) for label, prob in zip(labels, probabilities)}
        
        logger.info(f"Prediction: {prediction_label} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            prediction=prediction_label,
            confidence=confidence,
            probabilities=prob_dict,
            model_version=model_version
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.instances)
        
        # Make predictions
        predictions_idx = model.predict(features)
        probabilities = model.predict_proba(features)
        
        # Format results
        results = []
        for i, (pred_idx, probs) in enumerate(zip(predictions_idx, probabilities)):
            prediction_label = labels[pred_idx]
            confidence = float(probs[pred_idx])
            prob_dict = {label: float(prob) for label, prob in zip(labels, probs)}
            
            results.append({
                'prediction': prediction_label,
                'confidence': confidence,
                'probabilities': prob_dict
            })
        
        logger.info(f"Batch prediction: {len(results)} instances processed")
        
        return BatchPredictionResponse(
            predictions=results,
            model_version=model_version
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "n_features": model.n_features_in_,
        "classes": labels,
        "version": model_version
    }


if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)
