"""
ML Model Training Script
This script trains an Iris classification model and saves it to GCS.
"""

import argparse
import json
import logging
import os
import pickle
from datetime import datetime

import numpy as np
from google.cloud import storage
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data():
    """Load and prepare the Iris dataset."""
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Classes: {list(target_names)}")
    
    return X, y, feature_names, target_names


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest classifier."""
    logger.info(f"Training Random Forest with {n_estimators} estimators...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training completed!")
    
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, target_names):
    """Evaluate the trained model."""
    logger.info("Evaluating model...")
    
    # Training metrics
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Test metrics
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'test_f1_score': float(test_f1),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'timestamp': datetime.utcnow().isoformat()
    }
    
    logger.info(f"Training Accuracy: {train_accuracy:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    report = classification_report(y_test, test_pred, target_names=target_names)
    logger.info(f"\nClassification Report:\n{report}")
    
    return metrics


def save_model_locally(model, metrics, target_names, output_dir):
    """Save model and artifacts locally."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save class labels
    labels_path = os.path.join(output_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump({'labels': list(target_names)}, f, indent=2)
    logger.info(f"Labels saved to {labels_path}")
    
    return model_path, metrics_path, labels_path


def upload_to_gcs(local_dir, bucket_name, gcs_prefix):
    """Upload model artifacts to Google Cloud Storage."""
    logger.info(f"Uploading artifacts to gs://{bucket_name}/{gcs_prefix}/")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for filename in os.listdir(local_dir):
        local_path = os.path.join(local_dir, filename)
        blob_path = f"{gcs_prefix}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {filename} to gs://{bucket_name}/{blob_path}")
    
    return f"gs://{bucket_name}/{gcs_prefix}"


def main():
    parser = argparse.ArgumentParser(description='Train Iris classification model')
    parser.add_argument('--bucket', type=str, required=True,
                        help='GCS bucket name for storing artifacts')
    parser.add_argument('--output-prefix', type=str, default='models/iris',
                        help='GCS prefix for model artifacts')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in Random Forest')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size ratio')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Starting ML Model Training Pipeline")
    logger.info("=" * 50)
    
    # Load data
    X, y, feature_names, target_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = train_model(X_train, y_train, args.n_estimators, args.random_state)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, target_names)
    
    # Save locally
    local_output_dir = '/tmp/model_artifacts'
    save_model_locally(model, metrics, target_names, local_output_dir)
    
    # Upload to GCS
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    gcs_prefix = f"{args.output_prefix}/{timestamp}"
    artifact_uri = upload_to_gcs(local_output_dir, args.bucket, gcs_prefix)
    
    logger.info("=" * 50)
    logger.info("Training Pipeline Completed Successfully!")
    logger.info(f"Model artifacts: {artifact_uri}")
    logger.info("=" * 50)
    
    # Output for Vertex AI Pipelines
    print(f"ARTIFACT_URI={artifact_uri}")
    print(f"METRICS={json.dumps(metrics)}")


if __name__ == '__main__':
    main()
