# Automating MLOps Pipeline with Vertex AI and Cloud Deploy

## Hands-On Lab (45 Minutes) | Beginner-Medium Level

### Lab Overview

In this hands-on lab, you'll learn how to build an automated MLOps pipeline using Google Cloud's Vertex AI and Cloud Deploy. You'll create a complete end-to-end workflow that trains a machine learning model, containerizes it, and deploys it through multiple environments (staging → production) using Kubernetes namespaces on a single GKE cluster.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐   │
│   │  Source  │───▶│ Cloud Build  │───▶│  Artifact   │───▶│ Cloud Deploy │   │
│   │   Code   │    │  (CI/CD)     │    │  Registry   │    │  (CD)        │   │
│   └──────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘   │
│                          │                                       │          │
│                          ▼                                       ▼          │
│                   ┌──────────────┐                    ┌──────────────────┐  │
│                   │  Vertex AI   │                    │   GKE Cluster    │  │
│                   │  Pipelines   │                    │  ┌────────────┐  │  │
│                   └──────────────┘                    │  │  staging   │  │  │
│                          │                            │  │ namespace  │  │  │
│                          ▼                            │  ├────────────┤  │  │
│                   ┌──────────────┐                    │  │ production │  │  │
│                   │    Model     │                    │  │ namespace  │  │  │
│                   │   Registry   │                    │  └────────────┘  │  │
│                   └──────────────┘                    └──────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What You'll Learn

- ✅ Set up a Vertex AI training pipeline
- ✅ Build and containerize ML models
- ✅ Configure Cloud Deploy delivery pipelines
- ✅ Automate deployments with Cloud Build
- ✅ Implement progressive rollouts (staging → production) using namespaces
- ✅ Monitor and manage ML model deployments

### Prerequisites

- Google Cloud Platform account with billing enabled
- Basic knowledge of Python and Docker
- Familiarity with Kubernetes concepts
- Google Cloud SDK installed locally
- **Python 3.9 - 3.12** (recommended: use Cloud Shell which has Python pre-installed)

### Time Breakdown

| Section | Duration | Description |
|---------|----------|-------------|
| Setup | 5 min | Environment and project setup |
| Module 1 | 10 min | Create ML training code |
| Module 2 | 10 min | Build Vertex AI Pipeline |
| Module 3 | 10 min | Configure Cloud Deploy |
| Module 4 | 8 min | Deploy and test |
| Cleanup | 2 min | Resource cleanup |

---

## Initial Setup (5 minutes)

### Step 1: Set Environment Variables

```bash
# Set your project ID (auto-detect from gcloud config)
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export BUCKET_NAME="${PROJECT_ID}-mlops-lab"

# Or set manually if needed:
# export PROJECT_ID="your-project-id"

# Clone the repository
git clone https://github.com/misskecupbung/mlops-vertex-ai-cloud-deploy.git

# Set working directory
cd mlops-vertex-ai-cloud-deploy
```

### Step 2: Run Setup Script

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This script will:
- Enable required APIs
- Create a GCS bucket for artifacts
- Set up Artifact Registry repository
- Create a single GKE cluster with staging and production namespaces
- Configure Workload Identity for GCS access from pods

---

## Module 1: Create ML Training Code (10 minutes)

### Understanding the Model

We'll create a simple **Iris Flower Classification** model using scikit-learn. This model predicts the species of iris flowers based on their measurements.

### Step 1.1: Review the Training Script

Open `src/train.py` to understand the training logic:

```python
# Key components:
# 1. Load Iris dataset
# 2. Train a Random Forest classifier
# 3. Evaluate the model
# 4. Save model artifacts to GCS
```

### Step 1.2: Review the Serving Code

Open `src/serve.py` to understand the prediction API:

```python
# Key components:
# 1. Load model from artifacts
# 2. Expose REST API for predictions
# 3. Health check endpoints
```

### Step 1.3: Build Training Container

```bash
# Build the training image
gcloud builds submit --config=cloudbuild-training.yaml
```

### Step 1.4: Build Serving Container

```bash
# Build the serving image
gcloud builds submit --config=cloudbuild-serving.yaml
```

---

## Module 2: Build Vertex AI Pipeline (10 minutes)

### Understanding Vertex AI Pipelines

Vertex AI Pipelines help you automate, monitor, and govern your ML systems by orchestrating your ML workflow in a serverless manner.

### Step 2.1: Review Pipeline Definition

Open `src/pipeline.py` to understand the pipeline components:

```python
# Pipeline stages:
# 1. data_preparation - Load and validate data
# 2. model_training - Train the ML model
# 3. model_evaluation - Evaluate model metrics
# 4. model_upload - Upload to Model Registry
```

### Step 2.2: Compile the Pipeline

```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Compile the pipeline
python src/compile_pipeline.py
```

This generates `pipeline.json` - the compiled pipeline specification.

### Step 2.3: Submit Pipeline to Vertex AI

```bash
# Run the pipeline submission script
python src/submit_pipeline.py
```

### Step 2.4: Monitor Pipeline Execution

1. Go to [Vertex AI Pipelines Console](https://console.cloud.google.com/vertex-ai/pipelines)
2. Select your project
3. Click on the running pipeline to view progress
4. Wait for the pipeline to complete (approximately 5 minutes)

---

## Module 3: Configure Cloud Deploy (10 minutes)

### Understanding Cloud Deploy

Google Cloud Deploy is a managed, opinionated continuous delivery service that automates delivery of your applications to a series of target environments.

### Step 3.1: Review Cloud Deploy Configuration

Open `clouddeploy.yaml`:

```yaml
# Key components:
# - Delivery pipeline definition
# - Target environments (staging, production) using namespaces
# - Single GKE cluster with namespace isolation
# - Promotion rules
```

### Step 3.2: Review Kubernetes Manifests

Check the `k8s-manifests/` directory:
- `deployment.yaml` - Model serving deployment
- `service.yaml` - LoadBalancer service
- `hpa.yaml` - Horizontal Pod Autoscaler

### Step 3.3: Create Cloud Deploy Pipeline

```bash
# Get project number (needed for service account)
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')

# Substitute variables and render the config
envsubst < clouddeploy.yaml > clouddeploy-rendered.yaml

# Register the delivery pipeline
gcloud deploy apply \
  --file=clouddeploy-rendered.yaml \
  --region=${REGION}
```

### Step 3.4: Prepare ConfigMaps for Model Deployment

**IMPORTANT**: Create ConfigMaps with the model URI BEFORE creating a release.

```bash
# Create ConfigMap with model URI for staging
kubectl create configmap model-config \
  --from-literal=model_uri=gs://${PROJECT_ID}-mlops-lab/models/iris-classifier \
  -n staging \
  --dry-run=client -o yaml | kubectl apply -f -

# Create ConfigMap with model URI for production
kubectl create configmap model-config \
  --from-literal=model_uri=gs://${PROJECT_ID}-mlops-lab/models/iris-classifier \
  -n production \
  --dry-run=client -o yaml | kubectl apply -f -

echo "ConfigMaps created in staging and production namespaces"
```

### Step 3.5: Create Initial Release

```bash
# Create a release from the serving image
gcloud deploy releases create release-001 \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --delivery-pipeline=mlops-model-pipeline \
  --images=serving-image=${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-lab/serving:v1
```

---

## Module 4: Deploy and Test (8 minutes)

### Step 4.1: Monitor Staging Deployment

```bash
# Check release status
gcloud deploy releases describe release-001 \
  --delivery-pipeline=mlops-model-pipeline \
  --region=${REGION}

# List rollouts
gcloud deploy rollouts list \
  --delivery-pipeline=mlops-model-pipeline \
  --release=release-001 \
  --region=${REGION}
```

### Step 4.2: Configure Workload Identity for Staging

**IMPORTANT**: After Cloud Deploy creates the deployment, you must annotate the ServiceAccount for GCS access.

```bash
# Get cluster credentials (if not already done)
gcloud container clusters get-credentials mlops-cluster --region=${REGION}

# Wait for deployment to be created (check pods exist)
kubectl get pods -n staging

# Annotate ServiceAccount with Workload Identity
kubectl annotate serviceaccount model-serving-sa \
  --namespace staging \
  iam.gke.io/gcp-service-account=model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --overwrite

# Restart deployment to pick up the new identity
kubectl rollout restart deployment model-serving -n staging

# Wait for pods to be ready (1/1 Running)
kubectl get pods -n staging -w
```

### Step 4.3: Test Staging Endpoint

```bash
# Get the staging service IP
STAGING_IP=$(kubectl get svc model-serving -n staging -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test prediction endpoint
curl -X POST http://${STAGING_IP}:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Expected response:
```json
{
  "prediction": "setosa",
  "confidence": 0.95,
  "model_version": "v1"
}
```

### Step 4.4: Promote to Production

```bash
# Promote the release to production
gcloud deploy releases promote \
  --release=release-001 \
  --delivery-pipeline=mlops-model-pipeline \
  --region=${REGION}
```

Production deployments require approval. When prompted, type `Y` to confirm.

If the rollout shows `PENDING_APPROVAL`, approve it:

```bash
# Approve the production rollout
gcloud deploy rollouts approve release-001-to-production-0001 \
  --delivery-pipeline=mlops-model-pipeline \
  --release=release-001 \
  --region=${REGION}
```

### Step 4.5: Configure Workload Identity for Production

**IMPORTANT**: After the production deployment is created, annotate the ServiceAccount for GCS access.

```bash
# Wait for deployment to be created (check pods exist)
kubectl get pods -n production

# Annotate ServiceAccount with Workload Identity  
kubectl annotate serviceaccount model-serving-sa \
  --namespace production \
  iam.gke.io/gcp-service-account=model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --overwrite

# Restart deployment to pick up the new identity
kubectl rollout restart deployment model-serving -n production

# Wait for pods to be ready (1/1 Running)
kubectl get pods -n production -w
```

### Step 4.6: Test Production Endpoint

```bash
# Get production IP (same cluster, different namespace)
PROD_IP=$(kubectl get svc model-serving -n production -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test production endpoint
curl -X POST http://${PROD_IP}:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.7, 3.0, 5.2, 2.3]}'
```

### Step 4.7: View Deployment in Console

1. Go to [Cloud Deploy Console](https://console.cloud.google.com/deploy)
2. Click on `mlops-model-pipeline`
3. View the release progression through environments

### Step 4.8: Compare Both Environments

```bash
# View all deployments across namespaces
kubectl get deployments --all-namespaces | grep model-serving

# View all services
kubectl get svc --all-namespaces | grep model-serving
```

---

## Bonus: Automated CI/CD with Cloud Build (Optional)

### Set Up Continuous Deployment

The `cloudbuild.yaml` file defines the complete CI/CD workflow:

```bash
# Create Cloud Build trigger
gcloud builds triggers create github \
  --repo-name=your-repo \
  --repo-owner=your-github-username \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

Now, every push to `main` will:
1. Build training and serving containers
2. Run the Vertex AI pipeline
3. Create a new Cloud Deploy release
4. Automatically deploy to staging namespace

---

## Cleanup (2 minutes)

### Remove All Resources

```bash
chmod +x scripts/cleanup.sh
./scripts/cleanup.sh
```

This will delete:
- GKE cluster (with both namespaces)
- Cloud Deploy pipelines and releases
- Artifact Registry images
- GCS buckets
- Vertex AI pipeline runs

---

## Additional Resources

### Documentation
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Deploy Documentation](https://cloud.google.com/deploy/docs)
- [MLOps with Vertex AI](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Best Practices
- Use separate service accounts for training and serving
- Implement model versioning and A/B testing
- Set up monitoring and alerting for model drift
- Use network policies to isolate namespaces in production

### Next Steps
- Add model monitoring with Vertex AI Model Monitoring
- Implement canary deployments
- Add automated rollback on metric degradation
- Integrate with Vertex AI Feature Store

---

### Getting Help

- Check Cloud Logging for detailed error messages
- Review Vertex AI Pipeline logs in the console
- Examine GKE workload events with `kubectl describe -n <namespace>`

---

## Troubleshooting

### Common Issues

#### Pods stuck at 0/1 Ready with "No model found" error

**Symptom**: Pods are running but not ready, logs show "No model found"

```bash
kubectl logs -l app=model-serving -n staging
# Shows: WARNING - No model found. Server will start without model.
```

**Solution**: The ConfigMap is missing or has wrong value:

```bash
# Check current ConfigMap
kubectl get configmap model-config -n staging -o yaml

# Fix: Create/update ConfigMap with correct model URI
kubectl create configmap model-config \
  --from-literal=model_uri=gs://${PROJECT_ID}-mlops-lab/models/iris-classifier \
  -n staging \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart deployment
kubectl rollout restart deployment model-serving -n staging
```

#### Pods fail with "Permission denied" or "403 Forbidden" GCS error

**Symptom**: Pods crash with GCS permission errors

**Solution**: Workload Identity is not configured correctly:

```bash
# 1. Verify GCP service account exists
gcloud iam service-accounts list | grep model-serving-gsa

# 2. If not, create it and grant permissions
gcloud iam service-accounts create model-serving-gsa \
  --display-name="Model Serving Service Account" 2>/dev/null || echo "Already exists"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer" --quiet

# 3. Configure Workload Identity binding
gcloud iam service-accounts add-iam-policy-binding \
  model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[staging/model-serving-sa]" --quiet

# 4. Annotate K8s service account
kubectl annotate serviceaccount model-serving-sa \
  --namespace staging \
  iam.gke.io/gcp-service-account=model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --overwrite

# 5. Restart deployment
kubectl rollout restart deployment model-serving -n staging
```

#### Vertex AI Pipeline fails at model_training step

**Symptom**: Pipeline fails with "FileNotFoundError: data_artifact"

**Solution**: This is a GCS FUSE sync timing issue. Re-run the pipeline:

```bash
# Re-compile and re-submit the pipeline
python src/compile_pipeline.py
python src/submit_pipeline.py
```

#### Cloud Shell session expired

**Symptom**: `gcloud` commands fail with "no active account selected"

**Solution**: Re-authenticate and re-set environment variables:

```bash
gcloud auth login
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1
gcloud config set project ${PROJECT_ID}
gcloud auth application-default login
gcloud container clusters get-credentials mlops-cluster --region=${REGION}
```

#### LoadBalancer IP not assigned

**Symptom**: Service shows `<pending>` for EXTERNAL-IP

**Solution**: Wait a few minutes for GCP to provision the load balancer, or check quotas:

```bash
# Check service status
kubectl get svc model-serving -n staging -w

# If stuck, describe service for events
kubectl describe svc model-serving -n staging
```

---

*Lab created for educational purposes. Estimated cloud costs: ~$3-5 for the full lab (reduced with single cluster).*