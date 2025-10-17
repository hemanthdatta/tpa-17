# Getting Started with CLIP on Flowers102

## Quick Start Guide (3 Steps)

### Step 1: Install Dependencies
Open your terminal/PowerShell and run:

```powershell
cd c:\Users\heman\Desktop\tpa-17
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning)
- OpenCLIP (CLIP implementation)
- torchvision (datasets and transforms)
- matplotlib, seaborn (visualization)
- scikit-learn (metrics)

### Step 2: Run a Quick Test
Test that everything works:

```powershell
python quick_start.py
```

This will:
- Load a small CLIP model
- Download a sample of the Flowers dataset
- Extract features from a few images
- Show how image-text similarity works

### Step 3: Run Full Experiments

**Option A: Run Everything (Recommended)**
```powershell
python main.py --all
```

**Option B: Run Individual Experiments**
```powershell
# Zero-shot only
python main.py --zero-shot

# Linear probing only
python main.py --linear-probe
```

## What Will Happen?

### During Zero-Shot Evaluation:
1. Downloads CLIP model (~600MB, first time only)
2. Downloads Flowers102 dataset (~350MB, first time only)
3. Creates text embeddings for all 102 flower classes
4. Evaluates on train/val/test splits
5. Saves results and visualizations

**Time**: ~5-10 minutes

### During Linear Probing:
1. Uses the same CLIP model (already downloaded)
2. Extracts features from all images (one-time cost)
3. Trains a linear classifier for up to 50 epochs
4. Uses early stopping (typically stops around 20-30 epochs)
5. Saves trained model and results

**Time**: ~10-20 minutes

## Expected Output

### Console Output
You'll see progress bars and status messages:
```
============================================================
CLIP EXPERIMENTS ON FLOWERS102 DATASET
============================================================
Model: ViT-B-32
Pretrained: laion2b_s34b_b79k
============================================================

Loading CLIP model...
Dataset loaded successfully:
  Train: 1020 images
  Val: 1020 images
  Test: 6149 images

Running zero-shot evaluation...
Train Accuracy: 68.43%
Validation Accuracy: 67.25%
Test Accuracy: 68.91%

Running linear probing...
Epoch [1/50] Train Loss: 2.3451, Train Acc: 45.23%, Val Acc: 48.12%
...
Test Accuracy: 88.45%
```

### Generated Files
Check the `results/` folder:
```
results/
├── zero_shot_results.json                    # Your results!
├── zero_shot_confusion_matrix.png            # Visual analysis
├── linear_probe_results.json                 # Your results!
├── linear_probe_training_history.png         # Training curves
└── comparison_plot.png                       # Side-by-side comparison
```

## Customization

### Use a Different CLIP Model
```powershell
# Larger model (better accuracy, slower)
python main.py --all --model ViT-L-14 --pretrained laion2b_s32b_b82k

# Smaller model (faster, lower accuracy)
python main.py --all --model ViT-B-32-quickgelu
```

### Adjust Training Parameters
```powershell
# More epochs
python main.py --linear-probe --epochs 100

# Different learning rate
python main.py --linear-probe --lr 0.0001

# Larger batch size (if you have GPU memory)
python main.py --all --batch-size 64
```

## Troubleshooting

### "CUDA out of memory"
Your GPU doesn't have enough memory. Solutions:
```powershell
# Use smaller batch size
python main.py --all --batch-size 16

# Or use CPU (slower but works)
# Edit config.py and change:
# DEVICE = torch.device("cpu")
```

### "Dataset download failed"
Internet connection issue. The dataset will try to download automatically. If it fails:
1. Download manually from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
2. Extract to: `c:\Users\heman\Desktop\tpa-17\data\flowers-102\`

### "ModuleNotFoundError"
A package isn't installed:
```powershell
pip install [package_name]
# Or reinstall all:
pip install -r requirements.txt --force-reinstall
```

### Slow Performance
- **Use GPU**: Make sure CUDA is installed if you have an NVIDIA GPU
- **Reduce workers**: In config.py, set `NUM_WORKERS = 0` or `2`
- **Smaller dataset**: The test set is large (6149 images). Edit code to use a subset

## Understanding the Results

### Zero-Shot Accuracy (~65-75%)
- No training on Flowers dataset
- Pure transfer learning from CLIP's pretraining
- Same accuracy on train/val/test (no overfitting)

### Linear Probe Accuracy (~85-92%)
- Learns task-specific classifier
- Higher accuracy than zero-shot
- May show train > val > test (some overfitting)

### What's Good?
- **>80% test accuracy**: Excellent
- **70-80% test accuracy**: Good
- **<70% test accuracy**: Check your setup

## Next Steps

After running the basic experiments:

1. **Analyze Results**: Look at confusion matrices to see which flowers are hard
2. **Try Different Models**: Larger models usually give better accuracy
3. **Adjust Hyperparameters**: Learning rate, epochs, batch size
4. **Extend**: Implement fine-tuning or other methods from TPA17

## Need Help?

1. Check PROJECT_SUMMARY.md for detailed documentation
2. Look at the code comments - everything is documented
3. Run `python quick_start.py` to verify basic functionality
4. Check the course materials for TPA17

## Hardware Recommendations

### Minimum Requirements
- CPU: Any modern processor
- RAM: 8GB
- Storage: 2GB free space
- GPU: Optional (will use CPU if not available)

### Recommended Setup
- CPU: 4+ cores
- RAM: 16GB
- Storage: 5GB free space (SSD preferred)
- GPU: NVIDIA GPU with 6GB+ VRAM and CUDA support

## Estimated Times

### On CPU (e.g., Intel i7)
- Zero-shot: 15-20 minutes
- Linear probe: 30-40 minutes

### On GPU (e.g., RTX 3060)
- Zero-shot: 3-5 minutes
- Linear probe: 5-10 minutes

---

**You're all set! Run `python main.py --all` to get started.**

Good luck with your experiments! 🌸
