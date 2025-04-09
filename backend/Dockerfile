# Utilisation d'une image Python Alpine pour une image légère
FROM python:3.11.5

# Définir le répertoire de travail dans le conteneur
WORKDIR /backend

# Copier le fichier requirements.txt dans le conteneur depuis le répertoire backend/app
COPY backend/app/requirements.txt ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers du projet (code de l'app, etc.)
COPY backend/app/ ./app/

COPY backend/data/ ./data/

# Exposer le port 8000 pour l'application
EXPOSE 8000

# Commande pour démarrer l'application avec Uvicorn
CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

