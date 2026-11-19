# Wine Quality MLOps Pipeline (XGBoost + FastAPI + Docker + AWS ECS Fargate)

Pipeline MLOps de bout en bout pour prédire la qualité du vin (11 variables physico-chimiques) :
- **Modèle & suivi** : XGBoost + MLflow
- **API d'inférence** : FastAPI + Uvicorn (`POST /predict`)
- **Conteneurisation** : Docker (`python:3.10-slim`)
- **Cloud AWS** : Amazon ECR et Amazon ECS Fargate dans la région `eu-north-1` (selon la gouvernance du compte / SCP)

## Lancer en local avec Docker

```bash
docker build -t wine-mlops-api .
docker run -p 8000:8000 wine-mlops-api
## Architecture, Costs & MLOps Gaps

- **Stack & Region**: XGBoost, MLflow, FastAPI (`POST /predict`), Docker, ECR, and ECS Fargate in `eu-north-1`.
- **Cost Governance**: Monthly budget cap set to $5/month; ECS tasks are stopped when not actively testing to preserve credits.
- **Current MLOps Gaps**:
  - **CI/CD**: Basic syntax check in GitHub Actions; automated Docker build/push to ECR and deployment are not yet wired.
  - **Monitoring**: No continuous production telemetry, latency/error alerting, or drift tracking on the 11 input features.
  - **Retraining**: No automated data drift trigger or scheduled retraining pipeline yet.
