"""
Vertex AI Pipeline Definition
This script defines the ML training pipeline using Kubeflow Pipelines SDK.
"""

from kfp import dsl
from kfp.dsl import component, Input, Output, Artifact, Metrics, Model
from google.cloud import aiplatform


# Component 1: Data Preparation
@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.3.0", "pandas==2.0.3", "numpy==1.24.3"]
)
def data_preparation(
    test_size: float,
    random_state: int,
    data_artifact: Output[Artifact]
):
    """Load and prepare the Iris dataset."""
    import json
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save data info
    data_info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X.shape[1],
        'n_classes': len(iris.target_names),
        'feature_names': list(iris.feature_names),
        'target_names': list(iris.target_names)
    }
    
    # Save to artifact
    with open(data_artifact.path, 'w') as f:
        json.dump({
            'X_train': X_train.tolist(),
            'X_test': X_test.tolist(),
            'y_train': y_train.tolist(),
            'y_test': y_test.tolist(),
            'info': data_info
        }, f)
    
    print(f"Data prepared: {data_info}")


# Component 2: Model Training
@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.3.0", "numpy==1.24.3", "joblib==1.3.1"]
)
def model_training(
    data_artifact: Input[Artifact],
    n_estimators: int,
    random_state: int,
    model_artifact: Output[Model]
):
    """Train the Random Forest classifier."""
    import json
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    print("Loading training data...")
    with open(data_artifact.path, 'r') as f:
        data = json.load(f)
    
    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    
    print(f"Training model with {n_estimators} estimators...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Save model
    model_artifact.metadata['framework'] = 'sklearn'
    model_artifact.metadata['model_type'] = 'RandomForestClassifier'
    model_artifact.metadata['n_estimators'] = n_estimators
    
    joblib.dump(model, model_artifact.path)
    print("Model training completed!")


# Component 3: Model Evaluation
@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.3.0", "numpy==1.24.3", "joblib==1.3.1"]
)
def model_evaluation(
    data_artifact: Input[Artifact],
    model_artifact: Input[Model],
    metrics: Output[Metrics],
    accuracy_threshold: float
) -> bool:
    """Evaluate the trained model."""
    import json
    import joblib
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    print("Loading test data and model...")
    with open(data_artifact.path, 'r') as f:
        data = json.load(f)
    
    X_test = np.array(data['X_test'])
    y_test = np.array(data['y_test'])
    
    model = joblib.load(model_artifact.path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    metrics.log_metric('accuracy', accuracy)
    metrics.log_metric('f1_score', f1)
    metrics.log_metric('precision', precision)
    metrics.log_metric('recall', recall)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Check if model meets threshold
    passed = accuracy >= accuracy_threshold
    print(f"Accuracy threshold ({accuracy_threshold}): {'PASSED' if passed else 'FAILED'}")
    
    return passed


# Component 4: Model Upload
@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-aiplatform==1.36.0",
        "joblib==1.3.1",
        "google-cloud-storage==2.10.0"
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
    """Upload the model to Vertex AI Model Registry."""
    import json
    import joblib
    import pickle
    import os
    from google.cloud import aiplatform, storage
    
    if not model_passed_evaluation:
        print("Model did not pass evaluation. Skipping upload.")
        return ""
    
    print("Uploading model to Vertex AI Model Registry...")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Load model and data info
    model = joblib.load(model_artifact.path)
    with open(data_artifact.path, 'r') as f:
        data = json.load(f)
    
    # Create temporary directory for model artifacts
    artifact_dir = '/tmp/model_upload'
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Save model as pickle
    model_path = os.path.join(artifact_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save labels
    labels_path = os.path.join(artifact_dir, 'labels.json')
    with open(labels_path, 'w') as f:
        json.dump({'labels': data['info']['target_names']}, f)
    
    # Upload to GCS
    bucket_name = f"{project_id}-mlops-lab"
    gcs_prefix = f"models/{model_display_name}"
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for filename in os.listdir(artifact_dir):
        local_path = os.path.join(artifact_dir, filename)
        blob_path = f"{gcs_prefix}/{filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {filename}")
    
    artifact_uri = f"gs://{bucket_name}/{gcs_prefix}"
    
    # Upload model to Vertex AI
    vertex_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image,
        serving_container_environment_variables={
            'MODEL_URI': artifact_uri
        },
        labels={
            'framework': 'sklearn',
            'task': 'classification'
        }
    )
    
    model_resource_name = vertex_model.resource_name
    print(f"Model uploaded: {model_resource_name}")
    
    return model_resource_name


# Define the pipeline
@dsl.pipeline(
    name="iris-classification-pipeline",
    description="Train and deploy Iris classification model"
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
    """Main pipeline definition."""
    
    # Step 1: Data preparation
    data_prep_task = data_preparation(
        test_size=test_size,
        random_state=random_state
    )
    
    # Step 2: Model training
    training_task = model_training(
        data_artifact=data_prep_task.outputs['data_artifact'],
        n_estimators=n_estimators,
        random_state=random_state
    )
    
    # Step 3: Model evaluation
    evaluation_task = model_evaluation(
        data_artifact=data_prep_task.outputs['data_artifact'],
        model_artifact=training_task.outputs['model_artifact'],
        accuracy_threshold=accuracy_threshold
    )
    
    # Step 4: Model upload (conditional on evaluation)
    upload_task = model_upload(
        project_id=project_id,
        region=region,
        model_artifact=training_task.outputs['model_artifact'],
        data_artifact=data_prep_task.outputs['data_artifact'],
        model_display_name=model_display_name,
        serving_container_image=serving_container_image,
        model_passed_evaluation=evaluation_task.output
    )


if __name__ == '__main__':
    from kfp import compiler
    
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=iris_training_pipeline,
        package_path='pipeline.json'
    )
    print("Pipeline compiled to pipeline.json")
