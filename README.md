# Cloud Node Saturation MLOps

End-to-end MLOps pipeline for predicting cloud node saturation (`saturation_risk_score`) from infrastructure telemetry (`cpu_utilization_pct`, `memory_utilization_pct`, `request_rate_rps`, `p99_latency_ms`, `queue_depth`, `uptime_hours`).

![ML & MLOps Architecture](docs/mlops_architecture.svg)

---

## Key Highlights

- **Reproducible Training & Tracking**: Experiment tracking and model benchmark (`LinearRegression` baseline vs `XGBRegressor` candidate) with **MLflow** (`scripts/train_with_mlflow.py`), logging parameters, metrics (`RMSE`, `MAE`, `R²`), and model artifacts (`model/model.joblib`).
- **Data & Inference Validation**: Schema and dimension validation (`src/schema.py` and `scripts/app.py`) before training and serving.
- **Serving API**: Real-time inference API built with **FastAPI** (`scripts/app.py`) exposing `/health` and `/predict`.
- **Automated Quality Gates**: Automated test suite via **pytest** (`tests/`) and CI checks via **GitHub Actions** (`.github/workflows/`).
- **Cloud-Aware Deployment Design**: Configured with `DEPLOY_ENABLED=false`, GitHub OIDC, and scoped IAM roles (`PassRole` / `AssumeRole`) to handle organizational **AWS Service Control Policies (SCP)** restricting **Amazon ECR** and **ECS Fargate** in account `660178693695`.

---

## Repository Structure

```text
cloud-node-saturation-mlops/
├── .github/
│   └── workflows/             # GitHub Actions CI & CD workflows
├── docs/
│   └── mlops_architecture.svg # ML & MLOps architecture diagram
├── iam/                       # IAM trust & PassRole policies
├── scripts/
│   ├── app.py                 # FastAPI inference API (/health, /predict)
│   └── train_with_mlflow.py   # Baseline vs XGBoost & MLflow tracking
├── src/
│   └── schema.py              # Telemetry schema & validation
├── tests/                     # Pytest suite (unit, contract, model gate)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
