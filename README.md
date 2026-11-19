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