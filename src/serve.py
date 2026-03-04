"""FastAPI server for Iris classification model."""

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris Classification API",
    description="Serves predictions for Iris flower classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# globals
model = None
labels = None
model_version = os.getenv('MODEL_VERSION', 'v1')


class PredictionRequest(BaseModel):
    """Single prediction input."""
    features: List[float] = Field(
        ...,
        description="4 features: sepal_length, sepal_width, petal_length, petal_width",
        min_items=4,
        max_items=4,
        example=[5.1, 3.5, 1.4, 0.2]
    )


class BatchPredictionRequest(BaseModel):
    """Multiple predictions input."""
    instances: List[List[float]] = Field(
        ...,
        description="List of feature vectors",
        example=[[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


def download_model_from_gcs(bucket_name: str, prefix: str, local_dir: str = '/tmp/model'):
    """Pull model files from GCS to local dir."""
    logger.info(f"Downloading from gs://{bucket_name}/{prefix}/")
    
    os.makedirs(local_dir, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for blob in bucket.list_blobs(prefix=prefix):
        fname = os.path.basename(blob.name)
        if fname:
            blob.download_to_filename(os.path.join(local_dir, fname))
            logger.info(f"Downloaded {fname}")
    
    return local_dir


def load_model(model_dir: str):
    """Load pickle model and labels json."""
    global model, labels
    
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    with open(os.path.join(model_dir, 'labels.json'), 'r') as f:
        labels = json.load(f)['labels']
    
    logger.info(f"Loaded model, labels: {labels}")


@app.on_event("startup")
async def startup_event():
    """Load model on server start."""
    global model, labels
    
    model_uri = os.getenv('MODEL_URI', '')
    
    if model_uri.startswith('gs://'):
        parts = model_uri.replace('gs://', '').split('/', 1)
        bucket_name, prefix = parts[0], parts[1] if len(parts) > 1 else ''
        model_dir = download_model_from_gcs(bucket_name, prefix)
        load_model(model_dir)
    elif os.path.exists('/model'):
        load_model('/model')
    else:
        logger.warning("No model found - set MODEL_URI to load from GCS")


@app.get("/", response_model=dict)
async def root():
    """API info."""
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
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version
    )


@app.get("/ready")
async def readiness_check():
    """K8s readiness probe."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/live")
async def liveness_check():
    """K8s liveness probe."""
    return {"status": "alive"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single sample prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        X = np.array(request.features).reshape(1, -1)
        pred_idx = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        label = labels[pred_idx]
        conf = float(probs[pred_idx])
        prob_dict = {lbl: float(p) for lbl, p in zip(labels, probs)}
        
        logger.info(f"Predicted {label} ({conf:.3f})")
        
        return PredictionResponse(
            prediction=label,
            confidence=conf,
            probabilities=prob_dict,
            model_version=model_version
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple samples."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        X = np.array(request.instances)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        
        results = []
        for idx, (pred_idx, p) in enumerate(zip(preds, probs)):
            results.append({
                'prediction': labels[pred_idx],
                'confidence': float(p[pred_idx]),
                'probabilities': {lbl: float(v) for lbl, v in zip(labels, p)}
            })
        
        logger.info(f"Batch: {len(results)} predictions")
        
        return BatchPredictionResponse(predictions=results, model_version=model_version)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Model metadata."""
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
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
