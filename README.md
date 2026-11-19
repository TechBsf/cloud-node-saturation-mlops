# Wine Quality MLOps Pipeline (`eu-north-1`)

Production-minded MLOps pipeline for red wine quality prediction using **XGBoost**, **MLflow**, **FastAPI**, **Docker**, **Amazon ECR**, and **Amazon ECS Fargate**, built under a strict **5 $/month** cost guardrail in `eu-north-1`.

## Architecture & Stack
- **Model & Tracking**: `XGBoost` trained on the 11 wine features (`WineFeatures` schema validated with Pydantic in `src/schema.py`) and tracked with `MLflow` (`model/model.joblib`).
- **Serving API**: `FastAPI` exposed via `Uvicorn` on port `8000` (`POST /predict`) and packaged in a `Docker` image.
- **AWS Cloud (`eu-north-1`)**: Image stored in **Amazon ECR** and deployed on **Amazon ECS Fargate** for validation runs, while keeping active Fargate tasks scaled to `0` outside test windows.

## CI & Quality Gates
GitHub Actions (`.github/workflows/ci.yml`) runs automatically on `push` and `pull_request` to `main`:
- Python bytecode compilation (`python -m compileall .`)
- Automated unit & schema/model gate tests (`pytest -v` covering `src/schema.py` and `model/model.joblib`)
- Container build verification (`docker build`)

## Cost Governance ($5/Month Cap)
- **Zero idle compute**: ECS Fargate desired count is kept at `0` tasks when no test is running.
- **Budget discipline**: All cloud checks stay targeted in `eu-north-1` without continuous high-cost managed endpoints or always-on workers.

## Known MLOps Gaps & Next Steps
- **Continuous Drift & Observability**: No continuous production data-drift detector or centralized telemetry pipeline is active yet.
- **Automated Retraining & CD**: Model retraining and automated ECR/ECS rollout on model promotion are intentionally decoupled until data/model gates are expanded further.

## How a Reviewer Can Verify It Works

1. **Check automated CI runs**: Open the **Actions** tab on `TechBsf/wine-quality-mlops-aw` (`main`) to inspect `pytest` results (`src/schema.py` and `model/model.joblib`) and the `docker build` check.
2. **Run tests locally**: `pytest -v`
3. **Start the FastAPI service and test `POST /predict` (11 features)**: `uvicorn scripts.app:app --host 0.0.0.0 --port 8000` with sample payload `7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4` (HTTP `200` expected), while keeping ECS Fargate at `0` tasks in `eu-north-1` to stay within the /month budget.
