# MLOps Lab Quick Reference Cheatsheet

## Quick Start Commands

### Environment Setup
```bash
# Set environment variables (auto-detect project ID)
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export BUCKET_NAME="${PROJECT_ID}-mlops-lab"

# Or set manually if needed:
# export PROJECT_ID="your-project-id"

# Clone the repository
git clone https://github.com/misskecupbung/mlops-vertex-ai-cloud-deploy.git
cd mlops-vertex-ai-cloud-deploy

# Run setup
./scripts/setup.sh
```

### Build Containers
```bash
# Training container
gcloud builds submit --config=cloudbuild-training.yaml

# Serving container
gcloud builds submit --config=cloudbuild-serving.yaml

# With custom tag
gcloud builds submit --config=cloudbuild-training.yaml --substitutions=_TAG=v2
gcloud builds submit --config=cloudbuild-serving.yaml --substitutions=_TAG=v2
```

### Pipeline Commands
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Compile pipeline
python src/compile_pipeline.py

# Authenticate (required in Cloud Shell)
gcloud auth application-default login

# Submit pipeline
python src/submit_pipeline.py
```

### Cloud Deploy Commands
```bash
# Get project number (required for clouddeploy.yaml)
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')

# Render and apply delivery pipeline
envsubst < clouddeploy.yaml > clouddeploy-rendered.yaml
gcloud deploy apply --file=clouddeploy-rendered.yaml --region=${REGION}

# Create release
gcloud deploy releases create release-001 \
  --delivery-pipeline=mlops-model-pipeline \
  --region=${REGION} \
  --images=serving-image=${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-lab/serving:v1

# Promote to production
gcloud deploy releases promote \
  --release=release-001 \
  --delivery-pipeline=mlops-model-pipeline \
  --region=${REGION}

# Approve production rollout (required for production)
gcloud deploy rollouts approve release-001-to-production-0001 \
  --delivery-pipeline=mlops-model-pipeline \
  --release=release-001 \
  --region=${REGION}

# Check release status
gcloud deploy releases describe release-001 \
  --delivery-pipeline=mlops-model-pipeline \
  --region=${REGION}
```

### ConfigMap and Workload Identity Setup
```bash
# Create ConfigMap with model URI (do this for each namespace)
kubectl create configmap model-config \
  --from-literal=model_uri=gs://${PROJECT_ID}-mlops-lab/models/iris-classifier \
  -n staging \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap model-config \
  --from-literal=model_uri=gs://${PROJECT_ID}-mlops-lab/models/iris-classifier \
  -n production \
  --dry-run=client -o yaml | kubectl apply -f -

# Annotate service accounts for Workload Identity (after deployment creates them)
kubectl annotate serviceaccount model-serving-sa \
  --namespace staging \
  iam.gke.io/gcp-service-account=model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --overwrite

kubectl annotate serviceaccount model-serving-sa \
  --namespace production \
  iam.gke.io/gcp-service-account=model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --overwrite

# Restart deployments after annotation
kubectl rollout restart deployment model-serving -n staging
kubectl rollout restart deployment model-serving -n production
```

### Kubernetes Commands (Single Cluster with Namespaces)
```bash
# Get cluster credentials
gcloud container clusters get-credentials mlops-cluster --region=${REGION}

# Check deployments in staging
kubectl get deployments -n staging
kubectl get pods -n staging
kubectl get services -n staging

# Check deployments in production
kubectl get deployments -n production
kubectl get pods -n production
kubectl get services -n production

# Get staging service IP
kubectl get svc model-serving -n staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# Get production service IP
kubectl get svc model-serving -n production -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# View logs (staging)
kubectl logs -l app=model-serving -n staging -f

# View logs (production)
kubectl logs -l app=model-serving -n production -f

# View all resources across namespaces
kubectl get all --all-namespaces | grep -E "staging|production"
```

### Test Prediction API
```bash
# Get staging IP
STAGING_IP=$(kubectl get svc model-serving -n staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test staging health
curl http://${STAGING_IP}:8080/health

# Make staging prediction
curl -X POST http://${STAGING_IP}:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Get production IP
PROD_IP=$(kubectl get svc model-serving -n production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test production health
curl http://${PROD_IP}:8080/health

# Make production prediction
curl -X POST http://${PROD_IP}:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.7, 3.0, 5.2, 2.3]}'

# Batch prediction
curl -X POST http://${STAGING_IP}:8080/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}'
```

## Useful GCloud Commands

### Vertex AI
```bash
# List pipeline runs
gcloud ai custom-jobs list --region=${REGION}

# List models
gcloud ai models list --region=${REGION}

# Describe model
gcloud ai models describe MODEL_ID --region=${REGION}
```

### Cloud Deploy
```bash
# List pipelines
gcloud deploy delivery-pipelines list --region=${REGION}

# List releases
gcloud deploy releases list \
  --delivery-pipeline=mlops-model-pipeline \
  --region=${REGION}

# List rollouts
gcloud deploy rollouts list \
  --delivery-pipeline=mlops-model-pipeline \
  --release=release-001 \
  --region=${REGION}
```

### Artifact Registry
```bash
# List images
gcloud artifacts docker images list \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-lab

# Delete image
gcloud artifacts docker images delete \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-lab/serving:v1
```

## Troubleshooting

### Check API Status
```bash
gcloud services list --enabled | grep -E "aiplatform|clouddeploy|container"
```

### View Logs
```bash
# Cloud Build logs
gcloud builds log BUILD_ID

# GKE pod logs (staging)
kubectl logs -l app=model-serving -n staging --tail=100

# GKE pod logs (production)
kubectl logs -l app=model-serving -n production --tail=100

# Cloud Deploy logs
gcloud deploy rollouts describe ROLLOUT_NAME \
  --delivery-pipeline=mlops-model-pipeline \
  --release=release-001 \
  --region=${REGION}
```

### Common Fixes
```bash
# Restart pods in staging
kubectl rollout restart deployment model-serving -n staging

# Restart pods in production
kubectl rollout restart deployment model-serving -n production

# Check events in staging
kubectl get events -n staging --sort-by='.lastTimestamp'

# Check events in production
kubectl get events -n production --sort-by='.lastTimestamp'

# Verify namespaces exist
kubectl get namespaces

# Recreate namespace if needed
kubectl create namespace staging
kubectl create namespace production
```

### Pod Not Ready / Model Not Loading
```bash
# Check pod logs for errors
kubectl logs -l app=model-serving -n staging --tail=50

# Common issues:
# 1. "No model found" - ConfigMap model_uri is empty or wrong
kubectl get configmap model-config -n staging -o yaml

# 2. "403 Forbidden" - Workload Identity not configured
kubectl get serviceaccount model-serving-sa -n staging -o yaml
# Should have annotation: iam.gke.io/gcp-service-account

# 3. "FileNotFoundError: model.pkl" - Wrong model path
# Verify model exists in GCS:
gsutil ls -r gs://${PROJECT_ID}-mlops-lab/models/

# Fix ConfigMap with correct path:
kubectl create configmap model-config \
  --from-literal=model_uri=gs://${PROJECT_ID}-mlops-lab/models/iris-classifier \
  -n staging \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart deployment
kubectl rollout restart deployment model-serving -n staging
```

## Cleanup
```bash
./scripts/cleanup.sh
```

## Console Links

- [Vertex AI Pipelines](https://console.cloud.google.com/vertex-ai/pipelines)
- [Cloud Deploy](https://console.cloud.google.com/deploy)
- [GKE Workloads](https://console.cloud.google.com/kubernetes/workload)
- [Artifact Registry](https://console.cloud.google.com/artifacts)
- [Cloud Build](https://console.cloud.google.com/cloud-build)

## Architecture Notes

This lab uses a **single GKE cluster** with **namespace isolation**:

| Namespace | Purpose | Replicas |
|-----------|---------|----------|
| `staging` | Testing and validation | 1 |
| `production` | Live model serving | 2 |

Benefits of single cluster with namespaces:
- Reduced costs (one cluster instead of two)
- Faster setup time
- Easier resource management
- Namespace-level isolation for security
- Shared cluster resources with proper limits
