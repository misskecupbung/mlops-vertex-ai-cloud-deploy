#!/bin/bash
# Prepare namespaces with ConfigMaps and ServiceAccounts for Cloud Deploy

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Preparing namespaces${NC}"

[ -z "$PROJECT_ID" ] && export PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: PROJECT_ID not set${NC}"
    exit 1
fi

[ -z "$REGION" ] && export REGION="us-central1"

echo "PROJECT_ID=${PROJECT_ID}, REGION=${REGION}"

if [[ "$PROJECT_ID" == *" "* ]] || [[ "$PROJECT_ID" == "PROJECT_ID" ]]; then
    echo -e "${RED}Error: invalid PROJECT_ID '${PROJECT_ID}'${NC}"
    exit 1
fi

kubectl cluster-info &>/dev/null || gcloud container clusters get-credentials mlops-cluster --region=${REGION}

echo -e "${GREEN}[1/5] Setting up service account${NC}"
gcloud iam service-accounts describe model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com &>/dev/null || \
    gcloud iam service-accounts create model-serving-gsa --display-name="Model Serving SA"

echo -e "${GREEN}[2/5] Granting storage access${NC}"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer" \
    --condition=None --quiet 2>/dev/null || true

echo -e "${GREEN}[3/5] Configuring Workload Identity${NC}"
gcloud iam service-accounts add-iam-policy-binding \
    model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[staging/model-serving-sa]" --quiet 2>/dev/null || true

gcloud iam service-accounts add-iam-policy-binding \
    model-serving-gsa@${PROJECT_ID}.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="serviceAccount:${PROJECT_ID}.svc.id.goog[production/model-serving-sa]" --quiet 2>/dev/null || true

echo -e "${GREEN}[4/5] Configuring staging namespace${NC}"

kubectl delete configmap model-config -n staging --ignore-not-found
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: staging
data:
  model_uri: "gs://${PROJECT_ID}-mlops-lab/models/iris-classifier"
EOF

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

echo -e "${GREEN}[5/5] Configuring production namespace${NC}"

kubectl delete configmap model-config -n production --ignore-not-found
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: production
data:
  model_uri: "gs://${PROJECT_ID}-mlops-lab/models/iris-classifier"
EOF

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

echo ""
echo -e "${GREEN}Verification${NC}"

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

STAGING_URI=$(kubectl get configmap model-config -n staging -o jsonpath='{.data.model_uri}')
if [[ "$STAGING_URI" == *"PROJECT_ID"* ]]; then
    echo -e "${RED}Error: ConfigMap still has literal PROJECT_ID${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Done! You can now create a Cloud Deploy release:${NC}"
echo ""
echo "  gcloud deploy releases create release-001 \\"
echo "    --project=\${PROJECT_ID} \\"
echo "    --region=\${REGION} \\"
echo "    --delivery-pipeline=mlops-model-pipeline \\"
echo "    --images=serving-image=\${REGION}-docker.pkg.dev/\${PROJECT_ID}/mlops-lab/serving:v1"
