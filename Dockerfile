FROM python:3.12-slim

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT=8000

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements
COPY ./requirements.txt /app/requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . /app/

# Créer les dossiers nécessaires
RUN mkdir -p /app/reference_images

# Créer un utilisateur non-root
RUN useradd -m myuser && \
    chown -R myuser:myuser /app
USER myuser

# Commande par défaut
CMD uvicorn main:app --host 0.0.0.0 --port $PORT