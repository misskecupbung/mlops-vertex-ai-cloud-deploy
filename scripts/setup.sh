#!/bin/bash
# MLOps Lab Setup Script
# This script sets up all required GCP resources for the lab

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   MLOps Lab Setup Script              ${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}ERROR: PROJECT_ID environment variable is not set${NC}"
    echo "Please run: export PROJECT_ID=your-project-id"
    exit 1
fi

# Set default region if not set
REGION=${REGION:-us-central1}
BUCKET_NAME=${BUCKET_NAME:-${PROJECT_ID}-mlops-lab}

echo -e "${YELLOW}Configuration:${NC}"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Bucket: $BUCKET_NAME"
echo ""

# Set the project
echo -e "${GREEN}[1/7] Setting project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${GREEN}[2/7] Enabling required APIs...${NC}"
gcloud services enable \
    aiplatform.googleapis.com \
    cloudbuild.googleapis.com \
    clouddeploy.googleapis.com \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    compute.googleapis.com \
    iam.googleapis.com

echo "Waiting for APIs to be enabled..."
sleep 30

# Create GCS bucket for artifacts
echo -e "${GREEN}[3/7] Creating GCS bucket...${NC}"
if gsutil ls -b gs://${BUCKET_NAME} 2>/dev/null; then
    echo "Bucket already exists"
else
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
    echo "Bucket created: gs://${BUCKET_NAME}"
fi

# Create Artifact Registry repository
echo -e "${GREEN}[4/7] Creating Artifact Registry repository...${NC}"
if gcloud artifacts repositories describe mlops-lab --location=${REGION} 2>/dev/null; then
    echo "Repository already exists"
else
    gcloud artifacts repositories create mlops-lab \
        --repository-format=docker \
        --location=${REGION} \
        --description="MLOps Lab container images"
    echo "Repository created: mlops-lab"
fi

# Configure Docker authentication
echo -e "${GREEN}[5/7] Configuring Docker authentication...${NC}"
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# Create single GKE cluster
echo -e "${GREEN}[6/7] Creating GKE cluster...${NC}"
if gcloud container clusters describe mlops-cluster --region=${REGION} 2>/dev/null; then
    echo "Cluster already exists"
else
    gcloud container clusters create mlops-cluster \
        --region=${REGION} \
        --num-nodes=2 \
        --machine-type=e2-medium \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=5 \
        --workload-pool=${PROJECT_ID}.svc.id.goog
    echo "Cluster created: mlops-cluster"
fi

# Get cluster credentials and create namespaces
echo -e "${GREEN}[7/7] Setting up Kubernetes namespaces...${NC}"
gcloud container clusters get-credentials mlops-cluster --region=${REGION}

# Create staging namespace
kubectl create namespace staging --dry-run=client -o yaml | kubectl apply -f -
echo "Namespace created: staging"

# Create production namespace
kubectl create namespace production --dry-run=client -o yaml | kubectl apply -f -
echo "Namespace created: production"

# Add labels to namespaces for organization
kubectl label namespace staging environment=staging --overwrite
kubectl label namespace production environment=production --overwrite

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Setup Completed Successfully!       ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Resources created:"
echo "  - GCS Bucket: gs://${BUCKET_NAME}"
echo "  - Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-lab"
echo "  - GKE Cluster: mlops-cluster"
echo "  - Namespaces: staging, production"
echo ""
echo "Next steps:"
echo "1. Proceed to Module 1 in the README"
echo "2. Build the training and serving containers"
echo ""
echo -e "${YELLOW}Verify namespaces with: kubectl get namespaces${NC}"
