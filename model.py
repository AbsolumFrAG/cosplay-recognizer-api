import numpy as np
import tensorflow as tf
from keras.api.applications import MobileNetV3Large
from keras.api import layers, models, Sequential
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.applications.mobilenet_v3 import preprocess_input
from keras.api.utils import image_dataset_from_directory
from keras.api.optimizers import Adam
import time
from PIL import Image

class ModernCosplayClassifier:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.model = None
        self.class_names = None
        
        # Activer le GPU Metal
        tf.config.experimental.set_visible_devices(
            tf.config.list_physical_devices('GPU'), 'GPU'
        )

    def create_model(self, num_classes):
        """Crée un modèle basé sur MobileNetV3 avec transfer learning"""
        base_model = MobileNetV3Large(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        # Geler les couches de base
        base_model.trainable = False
        
        # Créer le modèle complet
        model = models.Sequential([
            base_model,
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model

    def create_augmentation_layer(self):
        """Crée une couche d'augmentation de données"""
        return Sequential([
            layers.RandomRotation(0.2),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomFlip("horizontal"),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ])

    def prepare_dataset(self, dataset_path, batch_size=32):
        """Prépare les datasets d'entraînement et de validation"""
        # Charger le dataset avec tf.data
        dataset = image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=self.image_size,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_dataset = image_dataset_from_directory(
            dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=self.image_size,
            batch_size=batch_size
        )

        self.class_names = dataset.class_names

        # Configurer le pipeline de données
        AUTOTUNE = tf.data.AUTOTUNE
        
        augmentation_layer = self.create_augmentation_layer()
        
        def prepare_for_training(ds, augment=True):
            # Mise en cache
            ds = ds.cache()
            
            if augment:
                ds = ds.map(
                    lambda x, y: (augmentation_layer(x, training=True), y),
                    num_parallel_calls=AUTOTUNE
                )

            # Prétraitement MobileNetV3
            ds = ds.map(
                lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE
            )
            
            # Préchargement
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            
            return ds

        train_ds = prepare_for_training(dataset)
        val_ds = prepare_for_training(val_dataset, augment=False)

        return train_ds, val_ds

    def train(self, dataset_path, batch_size=32, epochs=20):
        print("Préparation des données...")
        train_ds, val_ds = self.prepare_dataset(dataset_path, batch_size)
        
        num_classes = len(self.class_names)
        print(f"Création du modèle pour {num_classes} classes...")
        
        self.model = self.create_model(num_classes)
        
        # Compiler le modèle
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Entraînement initial
        print("Début de l'entraînement...")
        start_time = time.time()
        
        history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        initial_training_time = time.time() - start_time
        print(f"\nEntraînement initial terminé en {initial_training_time:.2f} secondes")
        
        # Fine-tuning
        print("\nDébut du fine-tuning...")
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Geler toutes les couches sauf les 20 dernières
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history_fine = self.model.fit(
            train_ds,
            epochs=10,
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        total_time = time.time() - start_time
        print(f"\nEntraînement total terminé en {total_time:.2f} secondes")
        
        # Sauvegarder
        self.model.save('modern_model.keras')
        np.save('class_names.npy', self.class_names)
        
        return history, history_fine
    
    def preprocess_image(self, image):
        """
        Prétraite une image pour la prédiction
        """
        if isinstance(image, str):
            image = tf.io.read_file(image)
            image = tf.image.decode_image(image, channels=3)
        elif isinstance(image, Image.Image):
            # Convertir l'image PIL en RGB et en array numpy
            image = image.convert('RGB')
            image = np.array(image)
        
        # Redimensionner l'image
        image = tf.image.resize(image, self.image_size)
        # S'assurer qu'on a 3 canaux
        image = tf.image.grayscale_to_rgb(image) if tf.shape(image)[-1] == 1 else image[:, :, :3]
        # Normaliser
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        return image

    @tf.function
    def predict_single(self, preprocessed_image):
        """Prédit sur une seule image prétraitée"""
        return self.model(preprocessed_image, training=False)

    def predict(self, image):
        """
        Prédit la classe d'une image
        """
        # Prétraiter l'image
        preprocessed = self.preprocess_image(image)
        # Ajouter la dimension du batch
        preprocessed = tf.expand_dims(preprocessed, 0)
        
        # Faire la prédiction
        predictions = self.predict_single(preprocessed)
        
        # Obtenir les 3 meilleures prédictions
        top_3_idx = tf.argsort(predictions[0])[-3:][::-1]
        results = []
        
        for idx in top_3_idx:
            idx = int(idx)
            results.append({
                "character": self.class_names[idx],
                "confidence": float(predictions[0][idx] * 100)
            })
        
        return results