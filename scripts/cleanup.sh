#!/bin/bash
# Cleanup script - removes all MLOps lab resources

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}MLOps Lab Cleanup${NC}"

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: PROJECT_ID not set${NC}"
    exit 1
fi

REGION=${REGION:-us-central1}
BUCKET_NAME=${BUCKET_NAME:-${PROJECT_ID}-mlops-lab}

echo -e "${YELLOW}Will delete: GKE cluster, Cloud Deploy pipeline, Artifact Registry, GCS bucket${NC}"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo
[[ ! $REPLY =~ ^[Yy]$ ]] && echo "Cancelled." && exit 0

echo -e "${GREEN}[1/5] Deleting Cloud Deploy pipeline${NC}"
gcloud deploy delivery-pipelines delete mlops-model-pipeline \
    --region=${REGION} --force --quiet 2>/dev/null || true

echo -e "${GREEN}[2/5] Deleting Cloud Deploy targets${NC}"
gcloud deploy targets delete staging --region=${REGION} --quiet 2>/dev/null || true
gcloud deploy targets delete production --region=${REGION} --quiet 2>/dev/null || true

echo -e "${GREEN}[3/5] Deleting GKE cluster${NC}"
gcloud container clusters delete mlops-cluster \
    --region=${REGION} --quiet --async 2>/dev/null || true

echo -e "${GREEN}[4/5] Deleting Artifact Registry${NC}"
gcloud artifacts repositories delete mlops-lab \
    --location=${REGION} --quiet 2>/dev/null || true

echo -e "${GREEN}[5/5] Deleting GCS bucket${NC}"
gsutil -m rm -r gs://${BUCKET_NAME} 2>/dev/null || true

echo ""
echo -e "${GREEN}Cleanup done!${NC}"
echo "Note: GKE deletion runs async. Check: gcloud container clusters list"
echo "Pipeline runs retained for audit: https://console.cloud.google.com/vertex-ai/pipelines"
