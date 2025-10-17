"""
Configuration file for CLIP experiments on Flowers dataset
"""

import torch

# Model Configuration
CLIP_MODEL = "ViT-B-32"  # CLIP model architecture
CLIP_PRETRAINED = "laion2b_s34b_b79k"  # Pre-trained weights

# Data Configuration
DATA_ROOT = "./data"
DATASET_NAME = "flowers102"
BATCH_SIZE = 32
NUM_WORKERS = 4

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
# Flower class names for Oxford Flowers 102 dataset
FLOWER_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", 
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura",
    "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata",
    "japanese anemone", "black-eyed susan", "silverbush", "californian poppy",
    "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia",
    "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss",
    "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia",
    "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"
]

# Text templates for zero-shot classification
TEXT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of a flower, a type of {}.",
    "a beautiful {}.",
    "{} flower.",
]
