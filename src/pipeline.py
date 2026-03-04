"""Vertex AI Pipeline for Iris model training and deployment."""

from kfp import dsl
from kfp.dsl import component, Input, Output, Artifact, Metrics, Model
from google.cloud import aiplatform


@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.3.2", "pandas==2.0.3", "numpy<2.0.0", "google-cloud-storage"]
)
def data_preparation(
    test_size: float,
    random_state: int,
    data_artifact: Output[Artifact]
):
    """Load Iris dataset and split into train/test."""
    import json
    import os
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    print("Loading Iris dataset")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    data_info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1],
        'n_classes': len(iris.target_names),
        'feature_names': list(iris.feature_names),
        'target_names': list(iris.target_names)
    }
    
    data_to_save = {
        'X_train': X_train.tolist(),
        'X_test': X_test.tolist(),
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist(),
        'info': data_info
    }
    
    print(f"Saving to {data_artifact.uri}")
    
    if data_artifact.uri.startswith('gs://'):
        from google.cloud import storage
        
        uri_parts = data_artifact.uri.replace('gs://', '').split('/', 1)
        bucket_name = uri_parts[0]
        blob_path = uri_parts[1] if len(uri_parts) > 1 else ''
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(json.dumps(data_to_save), content_type='application/json')
        print(f"Uploaded to GCS")
    else:
        artifact_dir = os.path.dirname(data_artifact.path)
        if artifact_dir:
            os.makedirs(artifact_dir, exist_ok=True)
        with open(data_artifact.path, 'w') as f:
            json.dump(data_to_save, f)
        print(f"Saved locally")
    
    print(f"Data: {data_info['train_samples']} train, {data_info['test_samples']} test")


@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.3.0", "numpy<2.0.0", "joblib==1.3.1", "google-cloud-storage"]
)
def model_training(
    data_artifact: Input[Artifact],
    n_estimators: int,
    random_state: int,
    model_artifact: Output[Model]
):
    """Train RandomForest on prepared data."""
    import json
    import os
    import io
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    print(f"Loading data from {data_artifact.uri}")
    
    if data_artifact.uri.startswith('gs://'):
        from google.cloud import storage
        
        uri_parts = data_artifact.uri.replace('gs://', '').split('/', 1)
        bucket_name = uri_parts[0]
        blob_path = uri_parts[1] if len(uri_parts) > 1 else ''
        
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        data = json.loads(blob.download_as_string())
    else:
        with open(data_artifact.path, 'r') as f:
            data = json.load(f)
    
    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    
    print(f"Training on {len(X_train)} samples, {n_estimators} trees")
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    model_artifact.metadata['framework'] = 'sklearn'
    model_artifact.metadata['model_type'] = 'RandomForestClassifier'
    model_artifact.metadata['n_estimators'] = n_estimators
    
    print(f"Saving model to {model_artifact.uri}")
    
    if model_artifact.uri.startswith('gs://'):
        from google.cloud import storage
        
        model_bytes = io.BytesIO()
        joblib.dump(clf, model_bytes)
        model_bytes.seek(0)
        
        uri_parts = model_artifact.uri.replace('gs://', '').split('/', 1)
        bucket_name = uri_parts[0]
        blob_path = uri_parts[1] if len(uri_parts) > 1 else ''
        
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        blob.upload_from_file(model_bytes, content_type='application/octet-stream')
    else:
        artifact_dir = os.path.dirname(model_artifact.path)
        if artifact_dir:
            os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(clf, model_artifact.path)
    
    print("Training complete")


@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.3.2", "numpy<2.0.0", "joblib==1.3.1", "google-cloud-storage"]
)
def model_evaluation(
    data_artifact: Input[Artifact],
    model_artifact: Input[Model],
    metrics: Output[Metrics],
    accuracy_threshold: float
) -> bool:
    """Evaluate model and check if it meets accuracy threshold."""
    import json
    import io
    import joblib
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    print(f"Loading test data and model")
    
    # load data
    if data_artifact.uri.startswith('gs://'):
        from google.cloud import storage
        uri_parts = data_artifact.uri.replace('gs://', '').split('/', 1)
        client = storage.Client()
        blob = client.bucket(uri_parts[0]).blob(uri_parts[1] if len(uri_parts) > 1 else '')
        data = json.loads(blob.download_as_string())
    else:
        with open(data_artifact.path, 'r') as f:
            data = json.load(f)
    
    X_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])
    
    # load model
    if model_artifact.uri.startswith('gs://'):
        from google.cloud import storage
        uri_parts = model_artifact.uri.replace('gs://', '').split('/', 1)
        client = storage.Client()
        blob = client.bucket(uri_parts[0]).blob(uri_parts[1] if len(uri_parts) > 1 else '')
        model = joblib.load(io.BytesIO(blob.download_as_bytes()))
    else:
        model = joblib.load(model_artifact.path)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    metrics.log_metric('accuracy', acc)
    metrics.log_metric('f1_score', f1)
    metrics.log_metric('precision', prec)
    metrics.log_metric('recall', rec)
    
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    passed = bool(acc >= accuracy_threshold)
    print(f"Threshold {accuracy_threshold}: {'PASS' if passed else 'FAIL'}")
    
    return passed


@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform>=1.38.0",
        "joblib==1.3.1",
        "google-cloud-storage>=2.10.0",
        "scikit-learn==1.3.2",
        "numpy<2.0.0"
    ]
)
def model_upload(
    project_id: str,
    region: str,
    model_artifact: Input[Model],
    data_artifact: Input[Artifact],
    model_display_name: str,
    serving_container_image: str,
    model_passed_evaluation: bool
) -> str:
    """Upload model to Vertex AI registry if evaluation passed."""
    import json
    import io
    import joblib
    import pickle
    import os
    from google.cloud import aiplatform, storage
    
    if not model_passed_evaluation:
        print("Evaluation failed, skipping upload")
        return ""
    
    print("Uploading to Vertex AI Model Registry")
    aiplatform.init(project=project_id, location=region)
    
    # load model from GCS
    if model_artifact.uri.startswith('gs://'):
        uri_parts = model_artifact.uri.replace('gs://', '').split('/', 1)
        client = storage.Client()
        blob = client.bucket(uri_parts[0]).blob(uri_parts[1] if len(uri_parts) > 1 else '')
        model = joblib.load(io.BytesIO(blob.download_as_bytes()))
    else:
        model = joblib.load(model_artifact.path)
    
    # load data info for labels
    if data_artifact.uri.startswith('gs://'):
        uri_parts = data_artifact.uri.replace('gs://', '').split('/', 1)
        client = storage.Client()
        blob = client.bucket(uri_parts[0]).blob(uri_parts[1] if len(uri_parts) > 1 else '')
        data = json.loads(blob.download_as_string())
    else:
        with open(data_artifact.path, 'r') as f:
            data = json.load(f)
    
    # save artifacts locally
    artifact_dir = '/tmp/model_upload'
    os.makedirs(artifact_dir, exist_ok=True)
    
    with open(os.path.join(artifact_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(artifact_dir, 'labels.json'), 'w') as f:
        json.dump({'labels': data['info']['target_names']}, f)
    
    # upload to GCS
    bucket_name = f"{project_id}-mlops-lab"
    gcs_prefix = f"models/{model_display_name}"
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for fname in os.listdir(artifact_dir):
        blob = bucket.blob(f"{gcs_prefix}/{fname}")
        blob.upload_from_filename(os.path.join(artifact_dir, fname))
    
    artifact_uri = f"gs://{bucket_name}/{gcs_prefix}"
    
    # register with Vertex AI
    vertex_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image,
        serving_container_environment_variables={'MODEL_URI': artifact_uri},
        labels={'framework': 'sklearn', 'task': 'classification'}
    )
    
    print(f"Model registered: {vertex_model.resource_name}")
    return vertex_model.resource_name


@dsl.pipeline(
    name="iris-classification-pipeline",
    description="Train and deploy Iris classifier"
)
def iris_training_pipeline(
    project_id: str,
    region: str,
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: int = 42,
    accuracy_threshold: float = 0.9,
    model_display_name: str = "iris-classifier",
    serving_container_image: str = ""
):
    """Full training pipeline: data prep -> train -> eval -> upload."""
    
    data_task = data_preparation(test_size=test_size, random_state=random_state)
    
    train_task = model_training(
        data_artifact=data_task.outputs['data_artifact'],
        n_estimators=n_estimators,
        random_state=random_state
    )
    
    eval_task = model_evaluation(
        data_artifact=data_task.outputs['data_artifact'],
        model_artifact=train_task.outputs['model_artifact'],
        accuracy_threshold=accuracy_threshold
    )
    
    model_upload(
        project_id=project_id,
        region=region,
        model_artifact=train_task.outputs['model_artifact'],
        data_artifact=data_task.outputs['data_artifact'],
        model_display_name=model_display_name,
        serving_container_image=serving_container_image,
        model_passed_evaluation=eval_task.outputs['Output']
    )


if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(pipeline_func=iris_training_pipeline, package_path='pipeline.json')
    print("Compiled to pipeline.json")
