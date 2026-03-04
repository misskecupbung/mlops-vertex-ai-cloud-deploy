"""Training script for Iris classifier. Trains a RandomForest model and uploads to GCS."""

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load Iris dataset and return features, labels, and metadata."""
    logger.info("Loading Iris dataset")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Classes: {list(iris.target_names)}")
    
    return X, y, iris.feature_names, iris.target_names


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train RandomForest with given hyperparameters."""
    logger.info(f"Training RandomForest ({n_estimators} trees)")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    logger.info("Training done")
    return clf


def evaluate_model(model, X_train, X_test, y_train, y_test, target_names):
    """Run evaluation metrics on train/test sets."""
    logger.info("Evaluating model")
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    metrics = {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'test_f1_score': float(test_f1),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'timestamp': datetime.utcnow().isoformat()
    }
    
    logger.info(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    logger.info(f"CV: {cv_scores.mean():.4f} +/- {cv_scores.std() * 2:.4f}")
    logger.info(f"\n{classification_report(y_test, test_pred, target_names=target_names)}")
    
    return metrics


def save_model_locally(model, metrics, target_names, output_dir):
    """Save model pickle, metrics json, and labels to local dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    labels_path = os.path.join(output_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump({'labels': list(target_names)}, f, indent=2)
    
    logger.info(f"Saved artifacts to {output_dir}")
    return model_path, metrics_path, labels_path


def upload_to_gcs(local_dir, bucket_name, gcs_prefix):
    """Upload all files from local_dir to GCS bucket."""
    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}"
    logger.info(f"Uploading to {gcs_uri}")
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for fname in os.listdir(local_dir):
        blob = bucket.blob(f"{gcs_prefix}/{fname}")
        blob.upload_from_filename(os.path.join(local_dir, fname))
    
    logger.info(f"Uploaded {len(os.listdir(local_dir))} files")
    return gcs_uri


def main():
    parser = argparse.ArgumentParser(description='Train Iris classifier')
    parser.add_argument('--bucket', type=str, required=True, help='GCS bucket for artifacts')
    parser.add_argument('--output-prefix', type=str, default='models/iris', help='GCS path prefix')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Starting training")
    logger.info("=" * 50)
    
    X, y, feature_names, target_names = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    logger.info(f"Split: {len(X_train)} train / {len(X_test)} test")
    
    model = train_model(X_train, y_train, args.n_estimators, args.random_state)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, target_names)
    
    local_dir = '/tmp/model_artifacts'
    save_model_locally(model, metrics, target_names, local_dir)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    artifact_uri = upload_to_gcs(local_dir, args.bucket, f"{args.output_prefix}/{timestamp}")
    
    logger.info("=" * 50)
    logger.info(f"Done! Artifacts at {artifact_uri}")
    logger.info("=" * 50)
    
    # stdout for pipeline parsing
    print(f"ARTIFACT_URI={artifact_uri}")
    print(f"METRICS={json.dumps(metrics)}")


if __name__ == '__main__':
    main()
