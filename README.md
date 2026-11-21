# Cloud Node Saturation MLOps

End-to-end MLOps pipeline for predicting cloud node saturation (`saturation_risk_score`) from infrastructure telemetry (`cpu_utilization_pct`, `memory_utilization_pct`, `request_rate_rps`, `p99_latency_ms`, `queue_depth`, `uptime_hours`).

![ML & MLOps Architecture](docs/mlops_architecture.svg)

---

## Key Highlights

- **Reproducible Training & Tracking**: baseline (`LinearRegression`) vs candidate (`XGBRegressor`) benchmarked with **MLflow** (`scripts/train_with_mlflow.py`), logging parameters, metrics (`RMSE`, `MAE`, `R²`) and model metadata.
- **Data & Inference Validation**: schema and dimension validation (`src/schema.py` and `scripts/app.py`) before training and serving. Feature order is defined once in `src/schema.py` and reused everywhere to avoid train/serve skew.
- **Serving API**: real-time inference API built with **FastAPI** (`scripts/app.py`) exposing health, readiness, metrics, drift and prediction endpoints.
- **Automated Quality Gates**: unit/contract/model-gate tests via **pytest** (`tests/`) plus lint (`ruff`), type-checking (`mypy`), training, and a **container smoke test** in CI (`.github/workflows/ci.yml`).
- **Cloud-Aware Deployment Design**: SageMaker training scripts (`run_training_job.py`, `create_pipeline.py`, `run_train_deploy.py`) plus an ECS Fargate CD workflow (`deploy.yml`) guarded by `DEPLOY_ENABLED` and GitHub OIDC, to handle organizational **AWS SCP** restrictions on **ECR** and **ECS Fargate**.
- **Deterministic Splits**: training, validation and test use the shared 70/15/15 helper with seed `42`; all benchmark scripts evaluate the same test split.
- **Model Promotion Gate**: only a candidate at least 1% better in validation RMSE replaces the linear baseline in `model/model.joblib`.
- **Training Data Validation**: missing values, non-numeric columns, infinite values and target values outside `[0,1]` fail before model training.
- **Model Manifest**: `model/model_metadata.json` records selected model, dataset hash, reference statistics, split, gate and test metrics.

---

## Model Results

The project trains a `LinearRegression` baseline and an `XGBRegressor` candidate on the same deterministic
70/15/15 split. A 5-fold cross-validation gate compares the mean validation RMSE.

| Model | Hold-out validation RMSE | Hold-out validation R² | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| LinearRegression | 0.0291 | 0.8968 | 0.0304 | 0.8737 |
| XGBRegressor | 0.0290 | 0.8976 | Not selected | Not selected |

**Selected model:** `LinearRegression`

The candidate improved the validation RMSE by less than the required 1%, so the promotion gate rejected it.
The linear model is simpler, more stable and was retained in `model/model.joblib`.

In the 5-fold gate, the linear baseline achieved mean RMSE `0.029277` and XGBoost
`0.030894`; the candidate also failed the cross-validation promotion criterion.

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/             # GitHub Actions CI & CD
├── docs/                      # Architecture diagrams
├── iam/                       # IAM trust & PassRole policies
├── scripts/
│   ├── app.py                 # FastAPI inference API (/health, /predict)
│   ├── train_with_mlflow.py   # Baseline vs XGBoost + MLflow tracking
│   ├── compute_baselines.py   # Offline benchmark table
│   ├── run_training_job.py    # SageMaker training job (SKLearn)
│   ├── create_pipeline.py     # SageMaker pipeline
│   └── run_train_deploy.py    # SageMaker built-in XGBoost train + deploy
├── src/
│   ├── schema.py              # Feature/TARGET contract + Pydantic model
│   ├── data/                  # CSV loading, validation and splitting
│   └── models/                # evaluation, inference, gates and drift
├── tests/                     # Pytest suite
├── data/                      # compute_capacity.csv
├── model/                     # trained model (not committed; train first)
├── Dockerfile                 # API image
├── requirements.txt           # runtime deps only
├── pyproject.toml             # packaging, deps, lint/type config
├── Makefile                   # common dev tasks
└── README.md
```

---

## Quickstart

### 1. Setup

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    # macOS/Linux : source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Train

```bash
python scripts/train_with_mlflow.py
```

This benchmarks `LinearRegression` against `XGBRegressor` on `data/compute_capacity.csv`,
writes the promoted model to `model/model.joblib`, and records its metadata.
Optionally enable MLflow tracking:

```bash
export MLFLOW_TRACKING_URI=file:///path/to/mlruns   # or your tracking server
python scripts/train_with_mlflow.py
```

### 3. Serve

```bash
uvicorn scripts.app:app --host 0.0.0.0 --port 8000
```

- `GET  /health` → `{"status": "ok"}`
- `POST /predict` with either a full telemetry object or a `{"features": [...]}` vector.
- `GET  /ready` → readiness status; returns `503` until the model is loaded.
- `GET  /metrics` → prediction count, error count, last score and average latency.
- `GET  /drift` → rolling feature drift report; becomes active after 30 predictions.

`/health` is a liveness endpoint: it confirms that the process responds. `/ready` is a readiness endpoint:
it confirms that the model was loaded successfully.

### 4. Test, lint, type-check

```bash
pytest -v
ruff check .
mypy .
```

Or use the Makefile targets: `make install`, `make train`, `make test`, `make lint`, `make typecheck`, `make api`.

### 5. Docker

```bash
docker build -t cloud-node-saturation-mlops .
docker run -p 8000:8000 cloud-node-saturation-mlops
```

> The model must exist locally (`make train`) before building, since `model/model.joblib` is not committed.

### 6. Deploy (SageMaker / ECS)

SageMaker scripts read configuration from the environment (`AWS_REGION`, `S3_BUCKET`, `SAGEMAKER_ROLE_ARN`); copy `.env.example` to `.env` and fill in your values.

The ECS CD workflow is off by default. To enable it, set these repository variables/secret:

- `DEPLOY_ENABLED=true`
- `DEPLOY_ROLE_ARN` (the OIDC role ARN used by `deploy.yml`)

---

## Known constraints

- The target AWS account restricts **Amazon ECR** and **ECS Fargate** via SCP; the ECS CD workflow stays disabled (`DEPLOY_ENABLED=false`) until that changes.
- The dataset is a small synthetic CSV (1500 rows), so the metrics demonstrate the pipeline rather than production performance.
- MLflow tracking is optional and may use local storage by default.
- The ECS and SageMaker deployments require AWS infrastructure and credentials and were not executed locally.
- Docker smoke testing could not be run locally until Docker Desktop/Linux containers are started.
- Drift monitoring is currently in memory and keeps only the 1,000 most recent values; production monitoring should persist observations and use a dedicated system.

## Technical Decisions

- The dataset uses a deterministic `70/15/15` train/validation/test split with seed `42`, keeping the test set isolated.
- Cross-validation is used to compare baseline and candidate models before promotion.
- The linear model was selected because XGBoost did not achieve the 1% validation RMSE improvement required by the gate.
- The feature contract in `src/schema.py` is shared by training and serving to reduce train/serve skew.
- GitHub Actions transfers the trained artifact between CI and CD instead of rebuilding an unrelated model image.
