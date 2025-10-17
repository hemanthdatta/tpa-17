# CLIP on Flowers102 - Project Summary

## Overview
This project implements CLIP (Contrastive Language-Image Pre-training) evaluation on the Oxford Flowers102 dataset, featuring two main approaches as per TPA17 requirements:

1. **Zero-Shot Evaluation** - Testing CLIP's ability to classify without training
2. **Linear Probing** - Training a linear classifier on frozen CLIP features

## Implementation Details

### Architecture
- **CLIP Model**: OpenCLIP implementation
- **Default Model**: ViT-B-32 with LAION-2B pre-training
- **Dataset**: Oxford Flowers102 (102 classes, ~8000 images)

### Key Components

#### 1. config.py
- Model configuration (architecture, pretrained weights)
- Dataset parameters (batch size, data paths)
- Training hyperparameters (learning rate, epochs, early stopping)
- 102 flower class names
- Text templates for zero-shot classification

#### 2. data_loader.py
- Loads Oxford Flowers102 using torchvision.datasets
- Provides train/validation/test splits
- Supports both standard ImageNet transforms and CLIP preprocessing
- Efficient DataLoader with multi-processing

#### 3. zero_shot.py
- ZeroShotClassifier class
- Creates text embeddings using multiple templates
- Computes image-text similarity for classification
- No training required - pure inference
- Evaluates on all splits (train/val/test)

#### 4. linear_probe.py
- LinearProbe class with frozen CLIP backbone
- Extracts features once, then trains linear layer
- Efficient batch training on extracted features
- Early stopping based on validation accuracy
- Model checkpointing

#### 5. utils.py
- Visualization functions (confusion matrix, training curves)
- Classification metrics (accuracy, precision, recall, F1)
- Result comparison plots
- JSON serialization for results
- Device information utilities

#### 6. main.py
- Command-line interface for experiments
- Orchestrates zero-shot and linear probing
- Generates comprehensive visualizations
- Saves all results and metrics
- Comparison between methods

## Technical Highlights

### Zero-Shot Classification
```python
# Key steps:
1. Load pre-trained CLIP model
2. Create text embeddings for all 102 classes using templates
3. For each image:
   - Extract image embedding
   - Compute similarity with all text embeddings
   - Predict class with highest similarity
```

**Templates used:**
- "a photo of a {}."
- "a photo of a flower, a type of {}."
- "a beautiful {}."
- "{} flower."

### Linear Probing
```python
# Key steps:
1. Load pre-trained CLIP model (frozen)
2. Extract image features for all samples (done once)
3. Train linear classifier:
   - Input: CLIP features (e.g., 512-dim)
   - Output: 102 classes
   - Optimizer: Adam with weight decay
   - Early stopping on validation accuracy
```

## Usage Examples

### Basic Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python main.py --all

# Individual experiments
python main.py --zero-shot
python main.py --linear-probe
```

### Advanced Usage
```bash
# Use larger model
python main.py --all --model ViT-L-14 --pretrained laion2b_s32b_b82k

# Adjust training parameters
python main.py --linear-probe --batch-size 64 --epochs 100 --lr 0.0001

# Quick test
python quick_start.py
```

## Expected Performance

### Zero-Shot Classification
- **Train**: 65-75%
- **Val**: 65-75%
- **Test**: 65-75%

Performance is consistent across splits since no training is involved.

### Linear Probing
- **Train**: 90-98%
- **Val**: 85-92%
- **Test**: 85-92%

Higher accuracy due to task-specific training, but potential for overfitting.

## Output Files

### Results Directory Structure
```
results/
├── zero_shot_results.json                    # Raw predictions and metrics
├── zero_shot_confusion_matrix.png            # Confusion matrix visualization
├── zero_shot_classification_report.txt       # Per-class metrics
├── linear_probe_results.json                 # Raw predictions and metrics
├── linear_probe_training_history.png         # Loss and accuracy curves
├── linear_probe_confusion_matrix.png         # Confusion matrix visualization
├── linear_probe_classification_report.txt    # Per-class metrics
├── linear_probe.pth                          # Trained model checkpoint
└── comparison_plot.png                       # Zero-shot vs linear probing
```

## Technical Specifications

### Dependencies
- **PyTorch**: Deep learning framework
- **open_clip_torch**: OpenCLIP implementation
- **torchvision**: Dataset and transforms
- **scikit-learn**: Metrics and evaluation
- **matplotlib/seaborn**: Visualization

### Computational Requirements
- **GPU**: Recommended (CUDA-capable)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~1GB for dataset, ~500MB for models

### Training Time (Approximate)
- **Zero-Shot**: 5-10 minutes (inference only)
- **Linear Probing**: 10-20 minutes (depending on epochs)

## Code Quality Features

1. **Modular Design**: Separate modules for each component
2. **Type Hints**: Clear function signatures
3. **Documentation**: Comprehensive docstrings
4. **Error Handling**: Graceful error messages
5. **Configurability**: Easy to modify parameters
6. **Reproducibility**: Fixed random seeds possible

## Future Extensions (Not Implemented Yet)

As per TPA17, additional methods could be implemented:

1. **Full Fine-tuning**: Unfreeze CLIP layers and train end-to-end
2. **Few-shot Learning**: K-shot evaluation with limited samples
3. **Adapter Layers**: Add trainable adapter modules
4. **Prompt Tuning**: Learn continuous prompts
5. **Feature Fusion**: Combine multiple CLIP models

## Debugging and Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
python main.py --all --batch-size 16  # Reduce batch size
```

**Slow Data Loading:**
- Modify `NUM_WORKERS` in config.py
- Use SSD for dataset storage

**Dataset Download Fails:**
- Check internet connection
- Manually download from official source

### Verification Script
```bash
# Test basic functionality
python quick_start.py

# Test data loading
python data_loader.py

# Check device info
python utils.py
```

## Citation and References

### CLIP Paper
```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
```

### OpenCLIP
```bibtex
@software{ilharco_gabriel_2021_5143773,
  author={Ilharco, Gabriel and Wortsman, Mitchell and others},
  title={OpenCLIP},
  year={2021},
  publisher={Zenodo},
}
```

## License
Educational use only. For research purposes.

## Contact
For questions or issues, please refer to the course TPA17 materials.

---
**Last Updated**: October 2025
**Project**: TPA-17 CLIP Implementation
**Status**: Complete (Zero-Shot & Linear Probing)
