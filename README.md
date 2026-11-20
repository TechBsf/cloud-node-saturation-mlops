# Cloud Compute Node Saturation Risk MLOps Pipeline

Production-minded MLOps pipeline predicting cloud compute node saturation risk (saturation_risk_score in [0.0, 1.0]) using **XGBoost**, **MLflow**, **FastAPI**, **Docker**, **Amazon ECR**, and **Amazon ECS Fargate**, backed by automated pytest quality gates.

## Architecture

- **Training & Tracking**: scripts/train_with_mlflow.py trains an XGBRegressor on data/compute_capacity.csv, logs metrics/artifacts in MLflow, and exports model/model.joblib.
- **Validation Contract**: src/schema.py validates ComputeNodeTelemetry (cpu_utilization_pct, memory_utilization_pct, equest_rate_rps, p99_latency_ms, queue_depth, uptime_hours).
- **Inference API**: scripts/app.py serves POST /predict via FastAPI and returns saturation_risk_score.
- **CI & Containerization**: .github/workflows/ci.yml runs pytest quality gates and docker build checks for Amazon ECR / Amazon ECS Fargate deployment.

## Local Commands

```powershell
.\venv\Scripts\python.exe scripts\train_with_mlflow.py
.\venv\Scripts\python.exe -m pytest -v
```

## Sample POST /predict Payload

```json
{
  "cpu_utilization_pct": 82.5,
  "memory_utilization_pct": 76.0,
  "request_rate_rps": 1450.0,
  "p99_latency_ms": 185.0,
  "queue_depth": 24,
  "uptime_hours": 312.0
}
```