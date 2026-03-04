#!/bin/bash
# Setup script for MLOps lab - creates GCP resources

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}MLOps Lab Setup${NC}"

# get project ID from gcloud if not set
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
        echo -e "${RED}Error: PROJECT_ID not set${NC}"
        echo "Run: export PROJECT_ID=your-project-id"
        exit 1
    fi
    export PROJECT_ID
fi

REGION=${REGION:-us-central1}
BUCKET_NAME=${BUCKET_NAME:-${PROJECT_ID}-mlops-lab}

echo -e "${YELLOW}Config: project=$PROJECT_ID, region=$REGION, bucket=$BUCKET_NAME${NC}"
echo ""

echo -e "${GREEN}[1/7] Setting project${NC}"
gcloud config set project $PROJECT_ID

echo -e "${GREEN}[2/7] Enabling APIs${NC}"
gcloud services enable \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    clouddeploy.googleapis.com \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    compute.googleapis.com \
    iam.googleapis.com

sleep 30

echo -e "${GREEN}[3/7] Creating GCS bucket${NC}"
if gsutil ls -b gs://${BUCKET_NAME} 2>/dev/null; then
    echo "Bucket exists"
else
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
fi

echo -e "${GREEN}[3.5/7] Setting bucket permissions${NC}"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format='value(projectNumber)')

gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.admin" --quiet 2>/dev/null || true

gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com" \
  --role="roles/storage.admin" --quiet 2>/dev/null || true

echo -e "${GREEN}[4/7] Creating Artifact Registry${NC}"
if gcloud artifacts repositories describe mlops-lab --location=${REGION} 2>/dev/null; then
    echo "Repository exists"
else
    gcloud artifacts repositories create mlops-lab \
        --repository-format=docker \
        --location=${REGION} \
        --description="MLOps Lab images"
fi

echo -e "${GREEN}[5/7] Configuring Docker auth${NC}"
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

echo -e "${GREEN}[6/7] Creating GKE cluster${NC}"
if gcloud container clusters describe mlops-cluster --region=${REGION} 2>/dev/null; then
    echo "Cluster exists"
else
    gcloud container clusters create mlops-cluster \
        --region=${REGION} \
        --num-nodes=1 \
        --machine-type=e2-medium \
        --disk-type=pd-standard \
        --disk-size=50GB \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=3 \
        --workload-pool=${PROJECT_ID}.svc.id.goog
fi

echo -e "${GREEN}[7/7] Setting up namespaces${NC}"
gcloud container clusters get-credentials mlops-cluster --region=${REGION}

kubectl create namespace staging --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace production --dry-run=client -o yaml | kubectl apply -f -

kubectl label namespace staging environment=staging --overwrite
kubectl label namespace production environment=production --overwrite

echo -e "${GREEN}Setting up Workload Identity${NC}"

gcloud iam service-accounts create model-serving-gsa \
  --display-name="Model Serving SA" 2>/dev/null || true

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer" \
  --condition=None --quiet

gcloud iam service-accounts add-iam-policy-binding \
  model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[staging/model-serving-sa]" --quiet

gcloud iam service-accounts add-iam-policy-binding \
  model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[production/model-serving-sa]" --quiet

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Created:"
echo "  - Bucket: gs://${BUCKET_NAME}"
echo "  - Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-lab"
echo "  - Cluster: mlops-cluster (staging, production namespaces)"
echo ""
echo "Next: proceed to Module 1 in the README"
echo -e "${YELLOW}Verify: kubectl get namespaces${NC}"
