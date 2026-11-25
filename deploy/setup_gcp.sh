#!/usr/bin/env bash
# One-time Google Cloud setup for continuous deployment of audit-copilot and invoice-autopilot
# to Cloud Run, using Workload Identity Federation (GitHub Actions gets short-lived tokens:
# no service-account key is ever created or stored).
#
# Prerequisites: a Google Cloud project with billing enabled, gcloud installed and logged in
# (gcloud auth login), and the GitHub CLI logged in (gh auth login).
#
# Usage: bash deploy/setup_gcp.sh <PROJECT_ID>
set -euo pipefail

PROJECT_ID="${1:?Usage: bash deploy/setup_gcp.sh <PROJECT_ID>}"
REGION="${REGION:-europe-west9}"              # Paris
OWNER="${GITHUB_OWNER:-nmc2112}"
REPOS=(audit-copilot invoice-autopilot)
AR_REPO="apps"
DEPLOYER="github-deployer"
RUNTIME="app-runtime"
POOL="github"
PROVIDER="github-provider"
SECRET="mistral-api-key"

DEPLOYER_SA="${DEPLOYER}@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_SA="${RUNTIME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "==> Project ${PROJECT_ID}, region ${REGION}"
gcloud config set project "${PROJECT_ID}" >/dev/null

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com iam.googleapis.com

echo "==> Artifact Registry repository '${AR_REPO}'"
gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "${AR_REPO}" --repository-format=docker --location="${REGION}" \
    --description="Container images deployed by GitHub Actions"

echo "==> Service accounts"
gcloud iam service-accounts describe "${DEPLOYER_SA}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${DEPLOYER}" --display-name="GitHub Actions deployer"
gcloud iam service-accounts describe "${RUNTIME_SA}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${RUNTIME}" --display-name="Cloud Run runtime (least privilege)"

echo "==> Waiting for the new service accounts to propagate"
for sa in "${DEPLOYER_SA}" "${RUNTIME_SA}"; do
  for _ in $(seq 1 12); do gcloud iam service-accounts describe "${sa}" >/dev/null 2>&1 && break; sleep 5; done
done
sleep 15

echo "==> Roles for the deployer (deploy services, push images, act as the runtime account)"
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" --member="serviceAccount:${DEPLOYER_SA}" \
    --role="${role}" --condition=None >/dev/null
done
gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_SA}" \
  --member="serviceAccount:${DEPLOYER_SA}" --role=roles/iam.serviceAccountUser >/dev/null

echo "==> Mistral API key in Secret Manager"
if ! gcloud secrets describe "${SECRET}" >/dev/null 2>&1; then
  gcloud secrets create "${SECRET}" --replication-policy=automatic
  read -r -s -p "Paste your Mistral API key (input hidden): " KEY; echo
  printf '%s' "${KEY}" | gcloud secrets versions add "${SECRET}" --data-file=-
  unset KEY
fi
gcloud secrets add-iam-policy-binding "${SECRET}" --member="serviceAccount:${RUNTIME_SA}" \
  --role=roles/secretmanager.secretAccessor >/dev/null

echo "==> Workload Identity Federation for GitHub (only repositories owned by ${OWNER})"
gcloud iam workload-identity-pools describe "${POOL}" --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "${POOL}" --location=global --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers describe "${PROVIDER}" --location=global \
  --workload-identity-pool="${POOL}" >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER}" --location=global \
    --workload-identity-pool="${POOL}" --display-name="GitHub OIDC" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
    --attribute-condition="assertion.repository_owner=='${OWNER}'"

POOL_NAME="$(gcloud iam workload-identity-pools describe "${POOL}" --location=global --format='value(name)')"
PROVIDER_NAME="$(gcloud iam workload-identity-pools providers describe "${PROVIDER}" --location=global \
  --workload-identity-pool="${POOL}" --format='value(name)')"

for repo in "${REPOS[@]}"; do
  gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA}" --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/${POOL_NAME}/attribute.repository/${OWNER}/${repo}" >/dev/null
done

echo "==> GitHub repository variables (not secrets: none of these values is sensitive)"
for repo in "${REPOS[@]}"; do
  gh variable set GCP_PROJECT_ID --repo "${OWNER}/${repo}" --body "${PROJECT_ID}"
  gh variable set GCP_REGION --repo "${OWNER}/${repo}" --body "${REGION}"
  gh variable set GCP_AR_REPO --repo "${OWNER}/${repo}" --body "${AR_REPO}"
  gh variable set GCP_WIF_PROVIDER --repo "${OWNER}/${repo}" --body "${PROVIDER_NAME}"
  gh variable set GCP_SERVICE_ACCOUNT --repo "${OWNER}/${repo}" --body "${DEPLOYER_SA}"
  gh variable set GCP_RUNTIME_SERVICE_ACCOUNT --repo "${OWNER}/${repo}" --body "${RUNTIME_SA}"
done

cat <<EOF

Done. Next: push to main (or run the CD workflow from the Actions tab).
The deploy job prints the public URL in the run summary.
Recommended: set a budget alert in Billing > Budgets & alerts, and a monthly limit in the Mistral console.
EOF
