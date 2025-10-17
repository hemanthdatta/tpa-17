# Troubleshooting Guide

This document covers common issues and their solutions when running CLIP fine-tuning on Flowers102.

## Table of Contents
- [AttributeError: 'LoRALinear' object has no attribute 'weight'](#lora-attribute-error)
- [CUDA Out of Memory](#cuda-oom)
- [Slow Training](#slow-training)
- [Low Accuracy](#low-accuracy)
- [Dataset Download Issues](#dataset-issues)
- [Import Errors](#import-errors)

---

## LoRA Attribute Error

### Problem
```
AttributeError: 'LoRALinear' object has no attribute 'weight'
```

### Cause
OpenCLIP's `MultiheadAttention` module directly accesses the `weight` attribute of its projection layers. When we replace these with `LoRALinear`, we need to expose the weight attribute properly.

### Solution ✅ (Fixed in latest version)
The `LoRALinear` class now exposes `weight` and `bias` as properties:

```python
@property
def weight(self):
    return self.linear.weight

@property
def bias(self):
    return self.linear.bias
```

Additionally, we now target MLP layers (`mlp.c_fc`, `mlp.c_proj`) instead of attention projections, which are safer to replace.

### Manual Fix (if using older version)
1. Pull latest changes: `git pull origin master`
2. Or update `adapter_finetuning.py` with the property definitions above

---

## CUDA Out of Memory

### Problem
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

### Solutions

#### 1. Reduce Batch Size
```bash
python main.py --lora --batch-size 16  # Default is 32
```

Or edit `config.py`:
```python
BATCH_SIZE = 16  # or even 8
```

#### 2. Use Gradient Accumulation
Edit the training loop to accumulate gradients:
```python
accumulation_steps = 4
for i, (images, labels) in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 3. Use Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 4. Use More Efficient Method
Switch from full fine-tuning to a parameter-efficient method:
- **BitFit**: Smallest memory footprint (~0.01% parameters)
- **LoRA**: Good balance (~0.1% parameters)
- **Prefix-tuning**: Medium (~0.5% parameters)

```bash
python main.py --bitfit  # Instead of --full-finetune
```

#### 5. Use Smaller Model
```bash
python main.py --model ViT-B-32  # Instead of ViT-L-14
```

---

## Slow Training

### Problem
Training takes too long or GPU utilization is low.

### Solutions

#### 1. Increase Batch Size (if memory allows)
```bash
python main.py --batch-size 64  # Default is 32
```

#### 2. Increase Number of Workers
Edit `config.py`:
```python
NUM_WORKERS = 4  # or 8, depending on your CPU cores
```

#### 3. Enable CUDA Benchmarking
Add to your training script:
```python
torch.backends.cudnn.benchmark = True
```

#### 4. Pin Memory
In `data_loader.py`, set:
```python
DataLoader(..., pin_memory=True)
```

#### 5. Use Mixed Precision
See CUDA OOM section above for implementation.

---

## Low Accuracy

### Problem
Model accuracy is lower than expected.

### Diagnosis Steps

#### 1. Check if Model is Training
Look at training loss - it should decrease:
```
Epoch 1: train_loss=4.5, val_loss=4.2
Epoch 2: train_loss=3.1, val_loss=3.0  ✓ Good
Epoch 3: train_loss=2.8, val_loss=2.9  ✓ Good
```

If loss is not decreasing, check learning rate.

#### 2. Check for Overfitting
```
Epoch 10: train_loss=0.5, train_acc=95%, val_acc=70%  ❌ Overfitting
```

Solutions:
- Increase weight decay: `WEIGHT_DECAY = 1e-3`
- Add dropout: Increase dropout in config
- Use data augmentation
- Early stopping (already implemented)

#### 3. Check for Underfitting
```
Epoch 50: train_acc=65%, val_acc=64%  ❌ Underfitting
```

Solutions:
- Increase model capacity (use LoRA instead of BitFit, or full fine-tuning)
- Decrease weight decay
- Increase learning rate
- Train for more epochs

### Expected Accuracies

| Method | Expected Test Accuracy |
|--------|----------------------|
| Zero-Shot | 65-75% |
| Linear Probe | 85-92% |
| BitFit | 82-87% |
| Prefix-tuning | 85-90% |
| LoRA | 88-93% |
| Full Fine-tune | 90-95% |

If you're getting significantly lower, check:
- Dataset is loading correctly
- Model is on GPU: `model.to(device)`
- Labels are correct (0-101 for 102 classes)

---

## Dataset Download Issues

### Problem
```
Error downloading Flowers102 dataset
```

### Solutions

#### 1. Check Internet Connection
The dataset downloads automatically from PyTorch servers.

#### 2. Manual Download
Download from [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/):
1. Download `102flowers.tgz`
2. Download `imagelabels.mat`
3. Download `setid.mat`
4. Extract to `./data/flowers-102/`

#### 3. Check Disk Space
The dataset requires ~500MB disk space.

#### 4. Use Different Data Directory
```bash
mkdir /path/to/large/disk/data
```

Edit `config.py`:
```python
DATA_DIR = "/path/to/large/disk/data"
```

---

## Import Errors

### Problem
```
ModuleNotFoundError: No module named 'open_clip'
```

### Solution
Install all dependencies:
```bash
pip install -r requirements.txt
```

### Specific Package Issues

#### OpenCLIP
```bash
pip install open-clip-torch
```

#### PyTorch (with CUDA)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Missing HuggingFace Hub
```bash
pip install huggingface-hub>=0.19.0
```

---

## Common Error Messages

### 1. "Expected tensor to be on device cuda:0 but got cpu"

**Solution**: Make sure to move all tensors to GPU:
```python
images = images.to(device)
labels = labels.to(device)
```

### 2. "The size of tensor a (X) must match the size of tensor b (Y)"

**Solution**: Check input image size. CLIP expects 224x224:
```python
transforms.Resize(224)
transforms.CenterCrop(224)
```

### 3. "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"

**Solution**: Check for NaN in logits before softmax:
```python
logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
```

### 4. "IndexError: Target X is out of bounds"

**Solution**: Flowers102 has 102 classes (0-101). Check:
```python
num_classes = 102  # Not 100!
```

---

## Performance Optimization Tips

### 1. Optimal Batch Size
Find the largest batch size that fits in memory:
```python
# Start with 32, double until OOM, then use the previous value
batch_sizes = [32, 64, 128, 256]
```

### 2. Learning Rate Tuning
Use learning rate finder:
```python
lrs = [1e-5, 1e-4, 1e-3, 1e-2]
# Run short training with each, pick the one with steepest loss decrease
```

### 3. Model Selection
- For quick experiments: ViT-B-32
- For best accuracy: ViT-L-14
- For limited memory: RN50

---

## Debugging Checklist

When something goes wrong, check:

- [ ] All packages installed: `pip list | grep -E "torch|clip"`
- [ ] GPU available: `torch.cuda.is_available()`
- [ ] Data loading correctly: `len(train_loader)`, `len(val_loader)`
- [ ] Model on GPU: `next(model.parameters()).device`
- [ ] Correct number of classes: `102`
- [ ] Learning rate not too high/low: `1e-5 to 1e-3`
- [ ] Loss is decreasing: Check training logs
- [ ] No NaN in loss: `torch.isnan(loss).any()`

---

## Getting Help

If you encounter an issue not covered here:

1. Check the error message carefully
2. Look at the full traceback
3. Search for similar issues in GitHub Issues
4. Check CLIP/OpenCLIP documentation
5. Print intermediate values to debug:
   ```python
   print(f"Images shape: {images.shape}")
   print(f"Labels shape: {labels.shape}")
   print(f"Model output shape: {outputs.shape}")
   ```

---

## Logging and Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Print model structure:
```python
print(model)
```

Count parameters:
```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
```

---

## Version Compatibility

Tested with:
- Python 3.8+
- PyTorch 2.0.0+
- OpenCLIP 2.20.0+
- CUDA 11.8+

If using different versions, some adjustments may be needed.
