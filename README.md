# Wine Quality MLOps Pipeline

Production-minded MLOps pipeline for red wine quality prediction using **XGBoost**, **MLflow**, **FastAPI**, **Docker**, **Amazon ECR**, and **Amazon ECS Fargate**, designed with automated quality gates and on-demand serverless execution.

## Architecture & Stack

* **Model & Tracking**: `XGBoost` trained on 11 physicochemical wine features (`WineFeatures` Pydantic schema validated in `src/schema.py`) and tracked with `MLflow` (`model/model.joblib`).
* **Serving API**: `FastAPI` served via `Uvicorn` on port `8000` (`POST /predict`) and packaged as a `Docker` image.
* **AWS Cloud (`eu-north-1`)**: Container image stored in **Amazon ECR** and deployed on **Amazon ECS Fargate** for validation runs, using a scale-to-zero configuration (`desiredCount = 0` in test windows) to eliminate idle compute costs.

## CI & Quality Gates

GitHub Actions (`.github/workflows/ci.yml`) runs automatically on every `push` and `pull_request` to `main`:

* Bytecode syntax verification (`python -m compileall .`)
* Automated unit, schema, and model artifact gate tests (`pytest -v` covering `src/schema.py` and `model/model.joblib`)
* Container image build verification (`docker build`)

## Quickstart & Verification

### 1. Inspect Automated CI
Open the **Actions** tab on `TechBsf/wine-quality-mlops-aw` (`main`) to review `pytest` checks (`src/schema.py` and `model/model.joblib`) and the `docker build` step.

### 2. Run Locally
```bash
pip install -r requirements.txt
pytest -v
uvicorn scripts.app:app --host 0.0.0.0 --port 8000
```

### 3. Test the Prediction Endpoint (`POST /predict`)
Send a JSON payload matching `WineFeatures` in `src/schema.py` to `http://localhost:8000/predict`.

## Roadmap & Future Improvements

* **Observability & Drift Detection**: Add structured JSON inference logging and automated data-drift checks on incoming payloads.
* **Automated CD & Retraining**: Trigger automated `ECR` push and `ECS Fargate` task definition updates on tagged releases (`v*`) via GitHub OIDC.
