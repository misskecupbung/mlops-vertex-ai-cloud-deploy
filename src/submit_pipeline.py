"""Submit compiled pipeline to Vertex AI."""

import os
from google.cloud import aiplatform


def submit_pipeline():
    project_id = os.getenv('PROJECT_ID')
    region = os.getenv('REGION', 'us-central1')
    bucket_name = os.getenv('BUCKET_NAME', f'{project_id}-mlops-lab')
    
    if not project_id:
        raise ValueError("PROJECT_ID env var required")
    
    print(f"Initializing Vertex AI ({project_id}, {region})")
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f'gs://{bucket_name}'
    )
    
    serving_image = f"{region}-docker.pkg.dev/{project_id}/mlops-lab/serving:v1"
    
    params = {
        'project_id': project_id,
        'region': region,
        'test_size': 0.2,
        'n_estimators': 100,
        'random_state': 42,
        'accuracy_threshold': 0.9,
        'model_display_name': 'iris-classifier',
        'serving_container_image': serving_image
    }
    
    print("Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    print("\nSubmitting job...")
    job = aiplatform.PipelineJob(
        display_name="iris-classification-training",
        template_path="pipeline.json",
        pipeline_root=f"gs://{bucket_name}/pipeline_root",
        parameter_values=params,
        enable_caching=True
    )
    job.submit()
    
    print(f"\nSubmitted: {job.display_name}")
    print(f"Resource: {job.resource_name}")
    print(f"\nView at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project_id}")
    return job


if __name__ == '__main__':
    submit_pipeline()
