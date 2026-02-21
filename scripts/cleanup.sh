#!/bin/bash
# MLOps Lab Cleanup Script
# This script removes all resources created during the lab

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${RED}========================================${NC}"
echo -e "${RED}   MLOps Lab Cleanup Script            ${NC}"
echo -e "${RED}========================================${NC}"

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}ERROR: PROJECT_ID environment variable is not set${NC}"
    echo "Please run: export PROJECT_ID=your-project-id"
    exit 1
fi

REGION=${REGION:-us-central1}
BUCKET_NAME=${BUCKET_NAME:-${PROJECT_ID}-mlops-lab}

echo -e "${YELLOW}This will delete the following resources:${NC}"
echo "  - GKE cluster: mlops-cluster (includes staging and production namespaces)"
echo "  - Cloud Deploy pipeline: mlops-model-pipeline"
echo "  - Artifact Registry repository: mlops-lab"
echo "  - GCS bucket: gs://${BUCKET_NAME}"
echo "  - Vertex AI pipeline runs"
echo ""

read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo -e "${GREEN}Starting cleanup...${NC}"

# Delete Cloud Deploy releases and pipeline
echo -e "${GREEN}[1/5] Deleting Cloud Deploy resources...${NC}"
gcloud deploy delivery-pipelines delete mlops-model-pipeline \
    --region=${REGION} \
    --force \
    --quiet 2>/dev/null || echo "Pipeline not found or already deleted"

# Delete Cloud Deploy targets
echo -e "${GREEN}[2/5] Deleting Cloud Deploy targets...${NC}"
gcloud deploy targets delete staging \
    --region=${REGION} \
    --quiet 2>/dev/null || echo "Staging target not found"

gcloud deploy targets delete production \
    --region=${REGION} \
    --quiet 2>/dev/null || echo "Production target not found"

# Delete GKE cluster (this also deletes all namespaces and workloads)
echo -e "${GREEN}[3/5] Deleting GKE cluster...${NC}"
gcloud container clusters delete mlops-cluster \
    --region=${REGION} \
    --quiet \
    --async 2>/dev/null || echo "Cluster not found"

# Delete Artifact Registry images and repository
echo -e "${GREEN}[4/5] Deleting Artifact Registry repository...${NC}"
gcloud artifacts repositories delete mlops-lab \
    --location=${REGION} \
    --quiet 2>/dev/null || echo "Repository not found"

# Delete GCS bucket
echo -e "${GREEN}[5/5] Deleting GCS bucket...${NC}"
gsutil -m rm -r gs://${BUCKET_NAME} 2>/dev/null || echo "Bucket not found"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Cleanup Complete!                   ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Note: GKE cluster deletion runs in the background and may take a few minutes."
echo "Check status with: gcloud container clusters list"
echo ""
echo "Vertex AI pipeline runs are retained for audit. Delete manually if needed at:"
echo "https://console.cloud.google.com/vertex-ai/pipelines"
