import os
import base64
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np
import logging
from keras.api.models import load_model
from model import ModernCosplayClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Classification de Cosplay")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "modern_model.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.npy")
REFERENCE_IMAGES_PATH = os.getenv("REFERENCE_IMAGES_PATH", "reference_images")
model = None
class_names = None
reference_images: Dict[str, str] = {}

def load_reference_images():
    """Charge et met en cache les images de référence en base64"""
    global reference_images
    
    for character in class_names:
        # Chercher l'image de référence (supporte jpg, jpeg, png)
        for ext in ['jpg', 'jpeg', 'png']:
            image_path = os.path.join(REFERENCE_IMAGES_PATH, f"{character}.{ext}")
            if os.path.exists(image_path):
                with open(image_path, "rb") as img_file:
                    # Lire et convertir l'image en base64
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode()
                    # Déterminer le type MIME
                    mime_type = f"image/{ext if ext != 'jpg' else 'jpeg'}"
                    reference_images[character] = f"data:{mime_type};base64,{img_base64}"
                break
        if character not in reference_images:
            logger.warning(f"Pas d'image de référence trouvée pour {character}")
            reference_images[character] = None

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'API"""
    global model, class_names
    try:
        logger.info("Chargement du modèle...")
        model = load_model(MODEL_PATH)
        class_names = np.load(CLASS_NAMES_PATH).tolist()
        logger.info(f"Modèle chargé avec succès. Classes disponibles : {len(class_names)}")
        
        # Charger les images de référence
        logger.info("Chargement des images de référence...")
        load_reference_images()
        logger.info(f"Images de référence chargées : {len(reference_images)} sur {len(class_names)}")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation : {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'initialisation")

classifier = ModernCosplayClassifier()
classifier.model = load_model('modern_model.keras')
classifier.class_names = np.load('class_names.npy').tolist()

@app.post("/predict")
async def predict_character(file: UploadFile = File(...)):
    """Prédit le personnage à partir d'une image"""
    try:
        # Vérifier l'extension du fichier
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail="Format de fichier non supporté. Utilisez JPG ou PNG."
            )

        # Lire l'image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Faire la prédiction
        results = classifier.predict(img)
        
        # Ajouter les images de référence aux résultats
        for result in results:
            result["reference_image"] = reference_images.get(result["character"])

        return JSONResponse(content={"predictions": results})

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))