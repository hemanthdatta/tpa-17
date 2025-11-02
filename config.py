"""
Configuration file for CLIP experiments on PlantVillage Crop Disease dataset
"""

import torch

# Model Configuration
CLIP_MODEL = "ViT-B-32"  # CLIP model architecture
CLIP_PRETRAINED = "laion2b_s34b_b79k"  # Pre-trained weights

# Data Configuration
DATA_ROOT = "./data"
DATASET_NAME = "plantvillage"  # Will be set dynamically
BATCH_SIZE = 32
NUM_WORKERS = 4

# Available Datasets Configuration
AVAILABLE_DATASETS = {
    "1": {
        "name": "plantvillage",
        "display_name": "PlantVillage Plant Disease",
        "path": "plantvillage/PlantVillage",
        "url": "https://www.kaggle.com/datasets/emmarex/plantdisease",
        "classes": None  # Will be auto-detected
    },
    "2": {
        "name": "neu_surface_defect",
        "display_name": "NEU Surface Defect Database",
        "path": "neu_surface_defect",
        "url": "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database",
        "classes": None  # Will be auto-detected
    },
    "3": {
        "name": "goldenhar_cfid",
        "display_name": "Goldenhar CFID",
        "path": "goldenhar_cfid",
        "url": "https://www.kaggle.com/datasets/isratjahan123/goldenhar-cfid",
        "classes": None  # Will be auto-detected
    },
    "4": {
        "name": "semiconductor_wafer",
        "display_name": "Multi-Class Semiconductor Wafer Image Dataset",
        "path": "semiconductor_wafer",
        "url": "https://www.kaggle.com/datasets/drtawfikrahman/multi-class-semiconductor-wafer-image-dataset",
        "classes": None  # Will be auto-detected
    },
    "5": {
        "name": "pcb_defect",
        "display_name": "PCB Defect (Modified)",
        "path": "pcb_defect",
        "url": "https://www.kaggle.com/datasets/breaddddd/pcb-defect-modified",
        "classes": None  # Will be auto-detected
    },
    "6": {
        "name": "aircraft",
        "display_name": "Aircraft Recognition (100 Variants)",
        "path": "aircraft",
        "url": "https://www.kaggle.com/datasets/aircraft",
        "classes": 100,  # 100 variant classes
        "num_train": 3334,
        "num_val": 3333,
        "num_test": 3333,
        "total_images": 10000,
        "use_csv": True,  # Uses CSV files for splits
        "csv_files": {
            "train": "train.csv",
            "val": "val.csv",
            "test": "test.csv"
        },
        "hierarchy": {
            "levels": ["Model", "Variant", "Family", "Manufacturer"],
            "num_variants": 100,
            "num_families": 70,
            "num_manufacturers": 41,
            "classification_level": "Variant"  # Using Variant (100 classes)
        }
    }
}

# Current dataset (will be set at runtime)
CURRENT_DATASET = None
CURRENT_DATASET_PATH = None

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
# Class names (will be auto-detected from dataset or set dynamically)
PLANT_DISEASE_CLASSES = []

# Text templates for zero-shot classification (generic for any dataset)
TEXT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of {}.",
    "an image of {}.",
    "{} in the image.",
    "a clear photo of a {}.",
]

# Dataset-specific text templates
DATASET_TEMPLATES = {
    "aircraft": [
        "a photo of a {} aircraft.",
        "a photo of a {} airplane.",
        "an image of a {} aircraft.",
        "a {} aircraft in flight.",
        "a clear photo of a {} aircraft.",
        "a side view of a {} aircraft.",
    ],
    "plantvillage": [
        "a photo of a {}.",
        "a leaf affected by {}.",
        "a plant with {}.",
        "symptoms of {}.",
        "a clear photo of {}.",
    ],
    "neu_surface_defect": [
        "a photo of {} defect.",
        "a surface with {}.",
        "an image of {} on metal.",
        "{} defect on steel surface.",
        "a clear image of {}.",
    ],
    "default": TEXT_TEMPLATES
}

def get_text_templates():
    """Get appropriate text templates for the current dataset"""
    if DATASET_NAME in DATASET_TEMPLATES:
        return DATASET_TEMPLATES[DATASET_NAME]
    return DATASET_TEMPLATES["default"]

