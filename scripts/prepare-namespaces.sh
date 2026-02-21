#!/bin/bash
# Script to prepare namespaces for Cloud Deploy
# This script creates ConfigMaps and ServiceAccounts with proper variable substitution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Preparing Namespaces for Deployment  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Validate environment variables
if [ -z "$PROJECT_ID" ]; then
    export PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
fi

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}ERROR: PROJECT_ID is not set${NC}"
    echo "Run: export PROJECT_ID=\$(gcloud config get-value project)"
    exit 1
fi

if [ -z "$REGION" ]; then
    export REGION="us-central1"
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  PROJECT_ID: ${PROJECT_ID}"
echo "  REGION: ${REGION}"
echo ""

# Validate PROJECT_ID format (should not contain spaces or be empty)
if [[ "$PROJECT_ID" == *" "* ]] || [[ "$PROJECT_ID" == "PROJECT_ID" ]]; then
    echo -e "${RED}ERROR: PROJECT_ID appears to be invalid: '${PROJECT_ID}'${NC}"
    exit 1
fi

# Check if GKE cluster credentials are configured
if ! kubectl cluster-info &>/dev/null; then
    echo -e "${YELLOW}Getting GKE cluster credentials...${NC}"
    gcloud container clusters get-credentials mlops-cluster --region=${REGION}
fi

# Create GCP service account for model serving (if not exists)
echo -e "${GREEN}[1/5] Setting up GCP Service Account...${NC}"
gcloud iam service-accounts describe model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null || \
    gcloud iam service-accounts create model-serving-gsa \
        --display-name="Model Serving Service Account"

# Grant storage access
echo -e "${GREEN}[2/5] Granting storage permissions...${NC}"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer" \
    --condition=None --quiet 2>/dev/null || true

# Configure Workload Identity bindings
echo -e "${GREEN}[3/5] Configuring Workload Identity bindings...${NC}"
gcloud iam service-accounts add-iam-policy-binding \
    model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[staging/model-serving-sa]" --quiet 2>/dev/null || true

gcloud iam service-accounts add-iam-policy-binding \
    model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[production/model-serving-sa]" --quiet 2>/dev/null || true

# Create resources for STAGING namespace
echo -e "${GREEN}[4/5] Configuring STAGING namespace...${NC}"

# Delete existing ConfigMap to ensure fresh creation with correct values
kubectl delete configmap model-config -n staging --ignore-not-found

# Create ConfigMap with actual PROJECT_ID (using heredoc to avoid variable issues)
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: staging
data:
  model_uri: "gs://${PROJECT_ID}-mlops-lab/models/iris-classifier"
EOF

# Create ServiceAccount with Workload Identity annotation
kubectl delete serviceaccount model-serving-sa -n staging --ignore-not-found
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: model-serving-sa
  namespace: staging
  annotations:
    iam.gke.io/gcp-service-account: "model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com"
EOF

# Create resources for PRODUCTION namespace
echo -e "${GREEN}[5/5] Configuring PRODUCTION namespace...${NC}"

# Delete existing ConfigMap to ensure fresh creation with correct values
kubectl delete configmap model-config -n production --ignore-not-found

# Create ConfigMap with actual PROJECT_ID
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: production
data:
  model_uri: "gs://${PROJECT_ID}-mlops-lab/models/iris-classifier"
EOF

# Create ServiceAccount with Workload Identity annotation
kubectl delete serviceaccount model-serving-sa -n production --ignore-not-found
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: model-serving-sa
  namespace: production
  annotations:
    iam.gke.io/gcp-service-account: "model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com"
EOF

# Verify the setup
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Verification                         ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo -e "${YELLOW}Staging ConfigMap:${NC}"
kubectl get configmap model-config -n staging -o jsonpath='{.data.model_uri}'
echo ""

echo -e "${YELLOW}Staging ServiceAccount:${NC}"
kubectl get serviceaccount model-serving-sa -n staging -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}'
echo ""

echo -e "${YELLOW}Production ConfigMap:${NC}"
kubectl get configmap model-config -n production -o jsonpath='{.data.model_uri}'
echo ""

echo -e "${YELLOW}Production ServiceAccount:${NC}"
kubectl get serviceaccount model-serving-sa -n production -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}'
echo ""

# Validate that PROJECT_ID was substituted correctly
STAGING_URI=$(kubectl get configmap model-config -n staging -o jsonpath='{.data.model_uri}')
if [[ "$STAGING_URI" == *"PROJECT_ID"* ]]; then
    echo ""
    echo -e "${RED}ERROR: ConfigMap still contains literal 'PROJECT_ID'${NC}"
    echo "Expected: gs://${PROJECT_ID}-mlops-lab/models/iris-classifier"
    echo "Got: ${STAGING_URI}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!                      ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "You can now create a Cloud Deploy release:"
echo ""
echo "  gcloud deploy releases create release-001 \\"
echo "    --project=\${PROJECT_ID} \\"
echo "    --region=\${REGION} \\"
echo "    --delivery-pipeline=mlops-model-pipeline \\"
echo "    --images=serving-image=\${REGION}-docker.pkg.dev/\${PROJECT_ID}/mlops-lab/serving:v1"
echo ""
