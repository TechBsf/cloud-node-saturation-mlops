# Utiliser une image Python officielle légère
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source et le modèle entraîné
COPY scripts/ scripts/
COPY model/ model/

# Exposer le port de l'API
EXPOSE 8000

# Commande de lancement par défaut de l'API avec Uvicorn
CMD ["uvicorn", "scripts.app:app", "--host", "0.0.0.0", "--port", "8000"]