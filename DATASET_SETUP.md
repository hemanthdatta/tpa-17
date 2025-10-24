# PlantVillage Dataset Setup Guide

## Overview
This project uses the **PlantVillage Crop Disease Dataset** to evaluate foundation model fine-tuning strategies in a low-data regime. The dataset contains images of healthy and diseased plant leaves across 38 classes.

## Dataset Information

### Classes (38 total)
The dataset includes diseases across multiple crops:
- **Apple**: scab, black rot, cedar rust, healthy
- **Blueberry**: healthy
- **Cherry**: powdery mildew, healthy
- **Corn**: Cercospora leaf spot, common rust, northern leaf blight, healthy
- **Grape**: black rot, Esca, leaf blight, healthy
- **Orange**: Haunglongbing (citrus greening)
- **Peach**: bacterial spot, healthy
- **Pepper**: bacterial spot, healthy
- **Potato**: early blight, late blight, healthy
- **Raspberry**: healthy
- **Soybean**: healthy
- **Squash**: powdery mildew
- **Strawberry**: leaf scorch, healthy
- **Tomato**: 10 different diseases/conditions including bacterial spot, early blight, late blight, leaf mold, etc.

### Download Instructions

#### Option 1: Kaggle Dataset (Recommended)
1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Download the dataset
3. Extract to: `./data/plantvillage/PlantVillage/`

#### Option 2: Manual Download
1. Visit the PlantVillage GitHub repository
2. Download the color images dataset
3. Organize in the following structure:

```
data/
└── plantvillage/
    └── PlantVillage/
        ├── Apple___Apple_scab/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── Apple___Black_rot/
        │   └── ...
        ├── Apple___Cedar_apple_rust/
        │   └── ...
        └── ... (other classes)
```

## Low-Data Regime Configuration

### Current Settings (config.py)
```python
SAMPLES_PER_CLASS = 50  # Number of training samples per class
USE_LIMITED_DATA = True  # Enable limited data mode
```

### Rationale
The project simulates **limited labeled data scenarios** common in specialized domains:
- **50 samples per class** provides ~1,900 total training images (from 38 classes)
- This represents a realistic low-data regime for domain-specific tasks
- Enables evaluation of fine-tuning efficiency with minimal labeled data

### Adjusting Data Regime
You can modify the low-data regime settings in `config.py`:

```python
# For even more limited data (extreme low-data regime)
SAMPLES_PER_CLASS = 20  # ~760 training images

# For moderate low-data regime
SAMPLES_PER_CLASS = 100  # ~3,800 training images

# To use full dataset (not recommended for this project)
USE_LIMITED_DATA = False
```

## Dataset Statistics

### Full Dataset (approximate)
- Total images: ~54,000
- Classes: 38
- Image format: RGB JPG
- Image size: Variable (will be resized to 224×224)

### Low-Data Regime (50 samples/class)
- Training: ~1,900 images
- Validation: ~11,400 images (15% of full data)
- Test: ~11,400 images (15% of full data)

## Data Augmentation

Training images use augmentation to improve generalization:
- Random crop (224×224)
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet stats)

Validation/test images use:
- Center crop (224×224)
- Normalization only

## Verification

After downloading, verify the dataset:

```bash
python data_loader.py
```

This will:
- Load the dataset
- Print dataset statistics
- Show sample batch information
- Confirm low-data regime is applied correctly

## Project Alignment

This dataset choice aligns with the project goals:

1. **Domain Shift**: Agricultural disease detection is a specialized domain distinct from the web-scale pre-training data used by foundation models (CLIP, DINOv2)

2. **Low-Data Regime**: Using 50 samples per class simulates realistic scenarios where labeled expert data is expensive

3. **Practical Application**: Plant disease classification is a real-world problem with significant impact in agriculture

4. **Evaluation Scope**: 38 classes provide sufficient complexity to evaluate fine-tuning strategies while remaining computationally feasible

## Troubleshooting

### Dataset Not Found Error
```
FileNotFoundError: PlantVillage dataset not found at ./data/plantvillage/PlantVillage
```

**Solution**: 
1. Download the dataset from Kaggle
2. Extract to the correct location
3. Ensure folder structure matches the expected format

### Class Count Mismatch
If you see fewer than 38 classes, ensure:
- All class folders are present
- Folder names match the expected format (e.g., `Apple___Apple_scab`)
- No empty folders exist

### Memory Issues
If you encounter memory issues with the full validation/test sets:
- Reduce batch size in `config.py`: `BATCH_SIZE = 16` or `BATCH_SIZE = 8`
- Reduce number of workers: `NUM_WORKERS = 2` or `NUM_WORKERS = 0`

## References

- PlantVillage Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease
- Original Paper: Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics.
