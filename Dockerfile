# Utiliser une image Python officielle légère
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Logs non bufferisés
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copier et installer les dépendances (couche de cache optimisée)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source (src requis par l'API) et le modèle entraîné
COPY src/ src/
COPY scripts/ scripts/
COPY model/ model/

# Exposer le port de l'API
EXPOSE 8000

# Ne pas exécuter en root
RUN useradd --create-home appuser
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]

# Commande de lancement par défaut de l'API avec Uvicorn
CMD ["uvicorn", "scripts.app:app", "--host", "0.0.0.0", "--port", "8000"]
