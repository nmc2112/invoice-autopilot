# Deployment (CD)

```
push to main ──► CI (lint, types, tests, eval, Docker build)
                   │ success
                   ▼
                 CD ──► publish image to ghcr.io/nmc2112/<repo>          (always)
                    └─► build → Artifact Registry → Cloud Run (Paris)    (once GCP is set up)
                         └─► smoke test /health, URL in the run summary
```

## What is already automatic
After every successful CI run on `main`, the image is published to **GitHub Container Registry**
(`ghcr.io/nmc2112/<repo>:latest` and `:<commit sha>`). No setup needed.

## One-time setup for Cloud Run (about 20 minutes)
1. Create a Google Cloud project and enable billing (Cloud Run has a free tier; set a budget alert).
2. Create a Mistral API key in the Mistral console and set a monthly spending limit.
3. Install the tools and log in:
   ```bash
   brew install --cask google-cloud-sdk
   gcloud auth login
   gh auth login
   ```
4. Run the setup script once (it serves both repositories):
   ```bash
   bash deploy/setup_gcp.sh <PROJECT_ID>
   ```
   It enables the APIs, creates an Artifact Registry repository, a deployer and a runtime service
   account, stores the Mistral key in Secret Manager (you type it, it is never written to disk),
   configures **Workload Identity Federation** (no service-account key) and sets the GitHub variables.
5. In GitHub, open Actions → CD → *Run workflow* (or push to `main`).

## Production choices for a public demo
- Region `europe-west9` (Paris): data stays in the EU.
- `--max-instances=1`, `--min-instances=0`: scales to zero, bounded cost.
- `RATE_LIMIT_PER_HOUR` per client IP protects the LLM budget.
- The API key is injected at runtime from Secret Manager; the runtime account can only read that secret.
- Storage is ephemeral (SQLite in the container): fine for a demo, use Cloud SQL for real data.
