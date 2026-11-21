# ☁️ Cloud Node Saturation — MLOps

> **Prédire la saturation d’un nœud cloud avant qu’il ne craque.**  
> Un pipeline MLOps complet, de la donnée brute à l’API de prédiction en temps réel, avec suivi d’expériences, tests automatisés et déploiement cloud.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.22-0194?logo=mlflow)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 De quoi s’agit-il ?

Une infrastructure cloud est composée de centaines (voire milliers) de serveurs virtuels. Quand l’un d’eux devient trop chargé — CPU, mémoire, requêtes par seconde, latence, file d’attente — les applications ralentissent ou s’arrêtent.

Ce projet construit un **modèle de machine learning** qui prédit un score de risque de saturation (`saturation_risk_score`, compris entre 0 et 1) à partir de six indicateurs d’infrastructure :

| Indicateur | Ce qu’il mesure |
|---|---|
| `cpu_utilization_pct` | Utilisation du processeur (%) |
| `memory_utilization_pct` | Utilisation de la mémoire (%) |
| `request_rate_rps` | Nombre de requêtes par seconde |
| `p99_latency_ms` | Latence du 99e percentile (ms) |
| `queue_depth` | Nombre de tâches en attente dans la file |
| `uptime_hours` | Temps de fonctionnement du nœud (heures) |

L’objectif : **anticiper la saturation** pour que les équipes ops puissent intervenir avant l’incident.

---

## 🛠️ Stack technique

| Couche | Technologie | Rôle |
|---|---|---|
| **Langage** | Python 3.10+ | Cœur du projet |
| **Données** | pandas, numpy | Chargement, nettoyage et split des données |
| **Modélisation** | scikit-learn, XGBoost | Régression linéaire vs boosting d’arbres |
| **Suivi d’expériences** | MLflow | Enregistrement des paramètres, métriques et modèles |
| **API de prédiction** | FastAPI + Uvicorn | Serving temps réel (`/predict`, `/health`, `/ready`, `/metrics`, `/drift`) |
| **Qualité du code** | pytest, ruff, mypy | Tests, lint et vérification des types |
| **Conteneurisation** | Docker | Image prête à déployer |
| **CI / CD** | GitHub Actions | Tests automatiques, build Docker, déploiement ECS |
| **Cloud** | AWS SageMaker, ECS, ECR, S3 | Entraînement et déploiement managés |
| **Sécurité** | IAM, OIDC, PassRole | Politiques d’accès minimal pour AWS |

---

## 🏗️ Comment ça marche ?

```
Données (CSV) → Validation → Split 70/15/15 → Entraînement → Modèle sélectionné → API FastAPI
                                                        ↑
                                          Comparaison Lin. Régression vs XGBoost
                                                        ↑
                                          Gate : le candidat doit améliorer le RMSE de ≥ 1 %
```

1. **Données** : le fichier `data/compute_capacity.csv` (1 500 lignes synthétiques) est chargé et validé (pas de valeurs manquantes, pas de valeurs aberrantes, cible entre 0 et 1).
2. **Entraînement** : un modèle de référence (`LinearRegression`) et un candidat (`XGBRegressor`) s’affrontent sur le même jeu de validation.
3. **Sélection** : seul le candidat qui améliore le RMSE d’au moins 1 % remplace la référence. Le résultat est enregistré avec ses métadonnées (`model_metadata.json`).
4. **Serving** : l’API FastAPI expose une endpoint `/predict` qui renvoie le score de saturation en temps réel.
5. **Surveillance** : l’API calcule une dérive de feature (drift) sur les 30 dernières prédictions.

---

## 🚀 Démarrage en 5 minutes

### 1. Prérequis

- **Python >= 3.10**
- (Optionnel) **Docker** pour construire l’image de l’API
- (Optionnel) Compte **AWS** pour le déploiement SageMaker / ECS

### 2. Installation

```bash
# Cloner le dépôt
git clone https://github.com/TechBsf/cloud-node-saturation-mlops.git
cd cloud-node-saturation-mlops

# Créer et activer l’environnement virtuel
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate

# Installer le projet et ses outils de développement
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Entraîner le modèle

```bash
python scripts/train_with_mlflow.py
```

Ce script compare la régression linéaire et XGBoost, applique la porte de sélection (≥ 1 % d’amélioration), et enregistre le meilleur modèle dans `model/model.joblib` ainsi que ses métadonnées dans `model/model_metadata.json`.

> Pour activer le suivi MLflow : `export MLFLOW_TRACKING_URI=file:///$(pwd)/mlruns` puis relancer l’entraînement.

### 4. Lancer l’API

```bash
uvicorn scripts.app:app --host 0.0.0.0 --port 8000
```

L’API est maintenant disponible sur `http://localhost:8000`.


## 📡 L’API en pratique

L’API expose deux façons d’envoyer une requête de prédiction.

### Format objet nommé (recommandé — plus lisible)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cpu_utilization_pct": 57.48,
    "memory_utilization_pct": 48.94,
    "request_rate_rps": 200.15,
    "p99_latency_ms": 32.16,
    "queue_depth": 4,
    "uptime_hours": 1602
  }'
```

**Réponse :**
```json
{ "saturation_risk_score": 0.3488 }
```

### Format vecteur brut

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [57.48, 48.94, 200.15, 32.16, 4, 1602]}'
```

### Autres endpoints utiles

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/health` | GET | Vérifie que le processus répond (liveness) |
| `/ready` | GET | Vérifie que le modèle est chargé (readiness) ; renvoie `503` sinon |
| `/metrics` | GET | Nombre de prédictions, erreurs, latence moyenne |
| `/drift` | GET | Détecte si les données entrantes s’écartent des données d’entraînement |

> Les bornes de chaque feature (ex. : le CPU doit être entre 0 et 100 %) sont déclarées une seule fois dans `src/schema.py` et partagées par l’entraînement et l’inférence. Pas de risque d’incohérence entre les deux.

---

## 🧪 Qualité du code

Le projet est couvert par une suite de tests automatisés et des outils de qualité :

```bash
# Lancer tous les tests (nécessite d’avoir entraîné le modèle au préalable)
make test

# Vérifier le style et les erreurs courantes
make lint

# Vérifier les annotations de type
make typecheck
```

**Résultat actuel :** ✅ **29 tests passés** (unitaires, contrat d’API, dérive de données, porte de sélection du modèle, etc.)

---

## 🐳 Docker

Construire et lancer l’image de l’API :

```bash
docker build -t cloud-node-saturation-mlops .
docker run -p 8000:8000 cloud-node-saturation-mlops
```

> ⚠️ Le modèle doit exister localement (`make train`) avant le build, car `model/model.joblib` n’est pas commit dans Git.

> 💡 **À améliorer :** le fichier `requirements.txt` utilisé par le `Dockerfile` doit contenir toutes les dépendances runtime de l’API (FastAPI, Uvicorn, Pydantic, Joblib, etc.) pour que l’image fonctionne correctement.

---

## ☁️ Déploiement Cloud (AWS)

Le projet est conçu pour s’exécuter sur AWS :

- **SageMaker** : scripts d’entraînement managés (`run_training_job.py`, `create_pipeline.py`, `run_train_deploy.py`).
- **ECS Fargate** : workflow de déploiement continu (CD) dans `.github/workflows/deploy.yml`, protégé par la variable `DEPLOY_ENABLED` et l’authentification OIDC GitHub.
- **IAM** : politiques minimales dans le dossier `iam/` (trust SageMaker + autorisation `iam:PassRole`).

> Le déploiement ECS est désactivé par défaut dans le compte cible (restrictions SCP sur ECR et ECS Fargate). Pour l’activer : `DEPLOY_ENABLED=true` + `DEPLOY_ROLE_ARN` en secret de dépôt.

## 📁 Organisation du dépôt

| Dossier / Fichier | Rôle |
|---|---|
| `src/schema.py` | Contrat de features et cible (source de vérité train/serve) |
| `src/data/` | Chargement, validation et split des données |
| `src/models/` | Évaluation, inférence, porte de sélection, dérive (drift) |
| `scripts/app.py` | API FastAPI (`/predict`, `/health`, `/ready`, `/metrics`, `/drift`) |
| `scripts/train_with_mlflow.py` | Entraînement, sélection de modèle, suivi MLflow |
| `scripts/run_training_job.py` / `create_pipeline.py` / `run_train_deploy.py` | Jobs et pipelines SageMaker |
| `tests/` | Suite pytest (unitaires, contrat, modèle, API) |
| `data/compute_capacity.csv` | Jeu de données synthétique (1 500 lignes) |
| `model/` | Artefact entraîné (`model.joblib`, non commit) + métadonnées |
| `.github/workflows/` | CI (tests, lint, typecheck, build Docker) et CD (ECS) |
| `iam/` | Politiques IAM AWS (trust + PassRole) |

---

## 📊 Résultats obtenus

Le pipeline entraîne une régression de référence et un candidat XGBoost sur le même split déterministe (70 % train / 15 % validation / 15 % test, seed 42).

| Modèle | RMSE validation | R² validation | RMSE test | R² test | Sélectionné ? |
|---|---:|---:|---:|---:|---:|
| `LinearRegression` | 0,0291 | 0,8968 | 0,0304 | 0,8737 | ✅ Oui |
| `XGBRegressor` | 0,0290 | 0,8976 | — | — | ❌ Non |

**Pourquoi la régression linéaire a été choisie ?**  
Le candidat XGBoost n’a pas amélioré le RMSE de validation suffisamment (≥ 1 % requis par la porte de sélection). Le modèle linéaire est plus simple, plus stable et plus facile à maintenir en production.

---

## 👨‍💻 Ce que ce projet démontre (pour un recruteur)

Ce dépôt illustre un cycle MLOps complet et professionnel :

- **Ingénierie des données** : validation stricte, split déterministe, gestion des valeurs aberrantes et manquantes.
- **Modélisation** : comparaison de plusieurs algorithmes, validation croisée, porte de sélection métrique, choix éclairé entre simplicité et performance.
- **Suivi d’expériences** : traçabilité complète avec MLflow (paramètres, métriques, artefacts).
- **Déploiement de modèle** : API REST temps réel avec gestion de la readiness, du drift et des métriques d’inférence.
- **Industrialisation** : tests automatisés (pytest), lint (ruff), typage statique (mypy), conteneurisation Docker, CI/CD GitHub Actions.
- **Cloud** : conception pour AWS (SageMaker, ECS, ECR, S3) avec gestion des identités (IAM, OIDC) et prise en compte des contraintes organisationnelles (SCP).

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier `LICENSE` pour plus de détails.

---

💡 *Besoin d’une démo ou d’une explication sur une partie du code ? N’hésitez pas à ouvrir une issue ou à me contacter.*