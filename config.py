"""
Configuration file for CLIP experiments on PlantVillage Crop Disease dataset
"""

import torch

# Model Configuration
CLIP_MODEL = "ViT-B-32"  # CLIP model architecture
CLIP_PRETRAINED = "laion2b_s34b_b79k"  # Pre-trained weights

# Data Configuration
DATA_ROOT = "./data"
DATASET_NAME = "plantvillage"
BATCH_SIZE = 32
NUM_WORKERS = 4

# Low-data regime configuration (for domain-specific fine-tuning)
# This simulates limited labeled data scenario as per project requirements
SAMPLES_PER_CLASS = 50  # Number of training samples per class (50-100 recommended for low-data regime)
USE_LIMITED_DATA = True  # Enable limited data mode

# Training Configuration (for Linear Probing)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Results Configuration
RESULTS_DIR = "./results"
SAVE_CHECKPOINT = True

# Zero-shot Configuration
# PlantVillage disease class names
PLANT_DISEASE_CLASSES = [
    "Apple scab",
    "Apple Black rot",
    "Apple Cedar apple rust",
    "Apple healthy",
    "Blueberry healthy",
    "Cherry Powdery mildew",
    "Cherry healthy",
    "Corn Cercospora leaf spot Gray leaf spot",
    "Corn Common rust",
    "Corn Northern Leaf Blight",
    "Corn healthy",
    "Grape Black rot",
    "Grape Esca (Black Measles)",
    "Grape Leaf blight (Isariopsis Leaf Spot)",
    "Grape healthy",
    "Orange Haunglongbing (Citrus greening)",
    "Peach Bacterial spot",
    "Peach healthy",
    "Pepper bell Bacterial spot",
    "Pepper bell healthy",
    "Potato Early blight",
    "Potato Late blight",
    "Potato healthy",
    "Raspberry healthy",
    "Soybean healthy",
    "Squash Powdery mildew",
    "Strawberry Leaf scorch",
    "Strawberry healthy",
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites Two-spotted spider mite",
    "Tomato Target Spot",
    "Tomato Tomato Yellow Leaf Curl Virus",
    "Tomato Tomato mosaic virus",
    "Tomato healthy"
]

# Text templates for zero-shot classification (agriculture/disease context)
TEXT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of a plant leaf with {}.",
    "a plant disease: {}.",
    "{} on plant leaves.",
    "agricultural crop with {}.",
]
