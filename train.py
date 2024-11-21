from model import ModernCosplayClassifier

if __name__ == "__main__":
    classifier = ModernCosplayClassifier()
    dataset_path = "dataset"  # Modifier selon votre structure
    classifier.train(dataset_path)