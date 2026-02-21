"""
Pipeline Submission Script
Submits the compiled pipeline to Vertex AI for execution.
"""

import os
from google.cloud import aiplatform


def submit_pipeline():
    # Get configuration from environment
    project_id = os.getenv('PROJECT_ID')
    region = os.getenv('REGION', 'us-central1')
    bucket_name = os.getenv('BUCKET_NAME', f'{project_id}-mlops-lab')
    
    if not project_id:
        raise ValueError("PROJECT_ID environment variable is required")
    
    # Initialize Vertex AI
    print(f"Initializing Vertex AI in project {project_id}, region {region}...")
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f'gs://{bucket_name}'
    )
    
    # Define pipeline parameters
    serving_image = f"{region}-docker.pkg.dev/{project_id}/mlops-lab/serving:v1"
    
    pipeline_params = {
        'project_id': project_id,
        'region': region,
        'test_size': 0.2,
        'n_estimators': 100,
        'random_state': 42,
        'accuracy_threshold': 0.9,
        'model_display_name': 'iris-classifier',
        'serving_container_image': serving_image
    }
    
    print("Pipeline parameters:")
    for key, value in pipeline_params.items():
        print(f"  {key}: {value}")
    
    # Create and submit pipeline job
    print("\nSubmitting pipeline job...")
    job = aiplatform.PipelineJob(
        display_name="iris-classification-training",
        template_path="pipeline.json",
        pipeline_root=f"gs://{bucket_name}/pipeline_root",
        parameter_values=pipeline_params,
        enable_caching=True
    )
    
    job.submit()
    
    print(f"\nPipeline job submitted!")
    print(f"Job name: {job.display_name}")
    print(f"Job resource name: {job.resource_name}")
    print(f"\nView pipeline run at:")
    print(f"https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project_id}")
    
    return job


if __name__ == '__main__':
    submit_pipeline()
