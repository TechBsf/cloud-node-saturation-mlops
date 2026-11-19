import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Wine Quality MLOps API")

# Charger le modèle entraîné (sauvegardé lors du run précédent)
# Ajustez le chemin selon l'emplacement de votre fichier .joblib ou artefact
model = joblib.load("model/model.joblib")  # ou le chemin vers votre artefact

class WineFeatures(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: WineFeatures):
    # Transformation en tableau numpy pour le modèle XGBoost
    features_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features_array)
    return {"prediction": float(prediction[0])}