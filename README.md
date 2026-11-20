# Cloud Node Saturation MLOps

End-to-end MLOps pipeline for predicting cloud node saturation (`saturation_score`) from infrastructure telemetry (`cpu_util`, `mem_util`, `net_io`, `disk_io`, `queue_depth`, `active_connections`).

---

## Key Highlights

- **Reproducible Training & Tracking**: Experiment tracking and model versioning with **MLflow** (`scripts/train_with_mlflow.py`), logging parameters, metrics (`RMSE`, `MAE`, `R²`), and artifacts.
- **Data & Inference Validation**: Schema and payload validation (`src/schema.py`) before training and serving.
- **Serving API**: Real-time inference endpoint built with **FastAPI** (`scripts/app.py`) and health check support.
- **Automated Quality Gates**: Unit and integration tests via **pytest** (`tests/`) and CI automation via **GitHub Actions** (`.github/workflows/`).
- **Cloud-Aware Deployment Design**: Configured with `DEPLOY_ENABLED=false` to explicitly handle organizational **AWS Service Control Policies (SCP)** restricting **Amazon ECR** and **ECS Fargate** pushes/deployments.

---

## Repository Structure

```text
cloud-node-saturation-mlops/
├── .github/workflows/       # GitHub Actions CI workflows
├── iam/                     # IAM policies / role definitions (if applicable)
├── mlartifacts/             # Local MLflow artifacts (generated/local)
├── scripts/
│   ├── app.py               # FastAPI inference application
│   └── train_with_mlflow.py # Training & MLflow tracking pipeline
├── src/
│   └── schema.py            # Input/output data schemas and validation
├── tests/                   # Test suite (pytest)
├── mlflow.db                # Local SQLite tracking backend (generated/local)
└── README.md
│   └── train_with_mlflow.py # Training & MLflow tracking pipeline
├── src/
│   └── schema.py            # Input/output data schemas and validation
├── tests/                   # Test suite (pytest)
├── mlflow.db                # Local SQLite tracking backend (generated/local)
└── README.md
