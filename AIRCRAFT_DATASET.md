# Aircraft Recognition Dataset Configuration

## Dataset Overview

The **Aircraft Recognition Dataset** contains 10,000 images of aircraft organized in a hierarchical classification system with 100 variant classes.

### Dataset Statistics

- **Total Images**: 10,000
- **Training Images**: 3,334
- **Validation Images**: 3,333
- **Test Images**: 3,333
- **Number of Classes**: 100 (Variant level)

### Hierarchical Structure

Aircraft models are organized in a four-level hierarchy:

1. **Model** (Finest level)
   - Example: Boeing 737-76J
   - Not used in evaluation (visually indistinguishable)

2. **Variant** (Classification level) ✓
   - Example: Boeing 737-700
   - **100 different variants**
   - Used for classification in this project

3. **Family**
   - Example: Boeing 737
   - 70 different families

4. **Manufacturer** (Coarsest level)
   - Example: Boeing
   - 41 different manufacturers

## Dataset Format

### CSV Structure

The dataset uses CSV files for easy data loading and splitting. Three separate CSV files are provided:

- `train.csv` - Training set (3,334 images)
- `val.csv` - Validation set (3,333 images)
- `test.csv` - Test set (3,333 images)

### CSV Columns

Each CSV file contains three columns:

| Column | Description | Example |
|--------|-------------|---------|
| `filename` | Image file name | `aircraft_001.jpg` |
| `Classes` | Variant class name | `707-320`, `A318` |
| `Labels` | Categorical value (0-99) | `0`, `1`, `2`, ... `99` |

### Example CSV Format

```csv
filename,Classes,Labels
aircraft_001.jpg,707-320,0
aircraft_002.jpg,A318,1
aircraft_003.jpg,737-700,2
...
```

## Dataset Setup

### Directory Structure

```
data/
└── aircraft/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── images/
        ├── aircraft_001.jpg
        ├── aircraft_002.jpg
        └── ...
```

### Download Instructions

1. Download the Aircraft Recognition dataset from Kaggle
2. Extract the dataset to `./data/aircraft/`
3. Ensure CSV files are in the root aircraft directory
4. Ensure all images are accessible from the paths in CSV files

### Verification

Run the following command to verify the dataset:

```bash
python show_datasets.py
```

Or test the data loader:

```bash
python data_loader.py
```

## Configuration Details

### Config.py Settings

The aircraft dataset has been added to `config.py` with the following configuration:

```python
"6": {
    "name": "aircraft",
    "display_name": "Aircraft Recognition (100 Variants)",
    "path": "aircraft",
    "url": "https://www.kaggle.com/datasets/aircraft",
    "classes": 100,
    "num_train": 3334,
    "num_val": 3333,
    "num_test": 3333,
    "total_images": 10000,
    "use_csv": True,
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
        "classification_level": "Variant"
    }
}
```

### Custom Data Loader

A custom `CSVDataset` class has been implemented in `data_loader.py` to handle CSV-based datasets:

- Reads CSV files with pandas
- Maps class names to indices automatically
- Handles image loading with error recovery
- Supports all standard transforms
- Compatible with low-data regime settings

## Running Experiments

### Select Aircraft Dataset

When running experiments, you can select the aircraft dataset:

```bash
python main.py --dataset 6 --all
```

Or use interactive mode:

```bash
python main.py
# Then select option 6 when prompted
```

### Specific Methods

Run specific fine-tuning methods on aircraft dataset:

```bash
# Zero-shot evaluation
python main.py --dataset 6 --zero-shot

# Linear probing
python main.py --dataset 6 --linear-probe

# LoRA fine-tuning
python main.py --dataset 6 --lora

# All adapter methods
python main.py --dataset 6 --adapters

# All experiments
python main.py --dataset 6 --all
```

### Low-Data Regime

The dataset supports low-data regime configuration. Adjust in `config.py`:

```python
SAMPLES_PER_CLASS = 50  # Default: 50 samples per class
USE_LIMITED_DATA = True  # Enable low-data mode
```

With 100 classes and 50 samples per class:
- **Training**: ~5,000 images (limited from 3,334)
- **Validation**: 3,333 images (full set)
- **Test**: 3,333 images (full set)

## Text Templates for Zero-Shot

Aircraft-specific text templates have been added for better zero-shot performance:

```python
DATASET_TEMPLATES = {
    "aircraft": [
        "a photo of a {} aircraft.",
        "a photo of a {} airplane.",
        "an image of a {} aircraft.",
        "a {} aircraft in flight.",
        "a clear photo of a {} aircraft.",
        "a side view of a {} aircraft.",
    ]
}
```

These templates help CLIP better understand aircraft classification context.

## Expected Performance

### Baseline Estimates

Based on similar fine-tuning projects with 100 classes:

| Method | Expected Accuracy | Trainable Parameters | Notes |
|--------|------------------|---------------------|--------|
| Zero-Shot | 30-45% | 0% | Baseline, domain shift |
| BitFit | 60-70% | ~0.01% | Quick adaptation |
| Linear Probe | 65-75% | <1% | Frozen features |
| Prefix-tuning | 70-78% | ~0.5% | Prompt learning |
| LoRA | 75-82% | ~0.1% | Best efficiency |
| Full Fine-tune | 80-88% | 100% | Maximum performance |

*Note: Actual results depend on hyperparameters, model architecture, and training configuration.*

### Challenges

1. **100 Classes**: More classes than typical datasets (38 in PlantVillage)
2. **Fine-grained**: Aircraft variants are visually similar
3. **Domain Shift**: CLIP trained on general web images, not aviation-specific
4. **Low-data Regime**: Limited samples per class (50 default)

### Strategies for Improvement

1. **Increase samples per class**: Set `SAMPLES_PER_CLASS = 100` in config
2. **Use larger models**: Try `--model ViT-L-14`
3. **Longer training**: Increase `NUM_EPOCHS = 100`
4. **Data augmentation**: Enable more augmentation in transforms
5. **Fine-tuning multiple layers**: Adjust LoRA rank or unfreeze more layers

## Class Distribution

The 100 variant classes include various manufacturers:

- **Boeing**: 737-700, 747-400, 777-200, etc.
- **Airbus**: A318, A320, A330, A380, etc.
- **Others**: Embraer, Bombardier, ATR, etc.

Full class list is auto-detected from the CSV files.

## Troubleshooting

### CSV File Not Found

**Error**: `CSV FILES NOT FOUND`

**Solution**:
1. Ensure CSV files are in the dataset root: `data/aircraft/`
2. Check filenames: `train.csv`, `val.csv`, `test.csv`
3. Verify paths in `config.py`

### Image Loading Errors

**Error**: `Error loading image ...`

**Solution**:
1. Check that filenames in CSV match actual image files
2. Ensure images are in the correct directory
3. Verify image paths are relative to dataset root
4. Check image file integrity (not corrupted)

### Memory Issues

**Error**: `CUDA out of memory`

**Solution**:
1. Reduce batch size: `--batch-size 16`
2. Use smaller model: `--model ViT-B-32`
3. Reduce workers: Set `NUM_WORKERS = 2` in config
4. Enable gradient checkpointing (for full fine-tuning)

### Low Accuracy

If accuracy is lower than expected:

1. **Disable low-data regime**: Set `USE_LIMITED_DATA = False`
2. **Increase training data**: Set `SAMPLES_PER_CLASS = 100`
3. **Try different learning rates**: `--lr 0.0001` to `--lr 0.01`
4. **Use larger model**: `--model ViT-L-14`
5. **Increase epochs**: `--epochs 100`
6. **Check class balance**: Ensure CSV files have balanced classes

## Integration with Existing Code

The aircraft dataset integrates seamlessly with the existing codebase:

### ✓ Supported Features

- ✅ Zero-shot evaluation
- ✅ Linear probing
- ✅ Full fine-tuning
- ✅ LoRA adaptation
- ✅ BitFit fine-tuning
- ✅ Prefix-tuning
- ✅ Low-data regime
- ✅ Confusion matrix visualization
- ✅ Training curves
- ✅ Classification reports
- ✅ Result saving

### Custom CSV Dataset Class

The `CSVDataset` class handles:
- Pandas DataFrame loading
- Class name to index mapping
- Image loading with error handling
- Transform application
- Compatibility with PyTorch DataLoader

### Automatic Detection

The data loader automatically detects CSV-based datasets:

```python
if dataset_info.get('use_csv', False):
    return get_csv_dataloaders(...)
```

No manual intervention needed - just select dataset 6!

## Future Enhancements

Potential improvements for aircraft dataset:

1. **Multi-level Classification**
   - Train models for all hierarchy levels
   - Family-level classification (70 classes)
   - Manufacturer-level classification (41 classes)

2. **Hierarchical Loss Functions**
   - Penalize errors based on hierarchy
   - Family-aware classification

3. **Data Augmentation**
   - Rotation (aircraft at different angles)
   - Color jittering (different lighting conditions)
   - Cutout/CutMix (occlusion handling)

4. **Multi-Task Learning**
   - Predict variant + family + manufacturer
   - Attribute prediction (engine count, size, etc.)

5. **Transfer Learning**
   - Pre-train on aviation-specific datasets
   - Domain adaptation techniques

## References

### Dataset
- Aircraft Recognition Dataset (Kaggle)
- FGVC-Aircraft Benchmark

### Related Papers
- Fine-Grained Visual Classification of Aircraft (BMVC 2013)
- Learning Fine-grained Image Similarity with Deep Ranking
- Bilinear CNN Models for Fine-grained Visual Recognition

---

**Ready to classify aircraft! ✈️🚁**
