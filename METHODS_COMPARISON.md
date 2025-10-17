# CLIP Fine-tuning Methods Comparison

This document provides a detailed comparison of all 6 implemented methods for evaluating and fine-tuning CLIP on the Flowers102 dataset.

## Overview

| Method | Type | Trainable Parameters | Training Time | Memory Usage | Accuracy (Expected) |
|--------|------|---------------------|---------------|--------------|---------------------|
| Zero-Shot | Evaluation | 0% | None | Low | 65-75% |
| Linear Probe | Feature-based | <1% (classifier only) | ~5 min | Low | 85-92% |
| BitFit | Parameter-efficient | ~0.01% (bias only) | ~10 min | Low | 82-87% |
| Prefix-tuning | Parameter-efficient | ~0.5% (prefix tokens) | ~15 min | Medium | 85-90% |
| LoRA | Parameter-efficient | ~0.1% (low-rank) | ~20 min | Medium | 88-93% |
| Full Fine-tune | Full model | 100% | ~60 min | High | 90-95% |

*Training times estimated on a single NVIDIA GPU (e.g., RTX 3090)*

## Detailed Method Analysis

### 1. Zero-Shot Evaluation

**What it does:**
- Uses pre-trained CLIP without any training on Flowers102
- Creates text embeddings for each class using multiple templates
- Classifies images by computing similarity with text embeddings

**When to use:**
- ✅ Quick baseline evaluation
- ✅ When you have no training data
- ✅ Evaluating model's general knowledge
- ❌ When you need high accuracy

**Pros:**
- No training required
- Extremely fast
- No GPU memory needed
- Good for understanding pre-trained model capabilities

**Cons:**
- Lower accuracy than trained methods
- Limited by pre-training quality
- Cannot adapt to domain-specific features

**Key Code Pattern:**
```python
# Create text features from class names
text_features = model.encode_text(text_tokens)

# Compute similarity
similarity = image_features @ text_features.T
predictions = similarity.argmax(dim=1)
```

---

### 2. Linear Probing

**What it does:**
- Freezes CLIP backbone completely
- Trains only a linear classifier on top of extracted features
- Two-stage process: extract features once, then train classifier

**When to use:**
- ✅ Limited computational resources
- ✅ When pre-trained features are already good
- ✅ Need for fast experimentation
- ❌ When domain shift is significant

**Pros:**
- Very efficient (features extracted once)
- Fast training
- Low memory requirements
- Good baseline for comparison

**Cons:**
- Cannot adapt backbone to new domain
- Limited by frozen features
- May underperform on domain-specific tasks

**Key Code Pattern:**
```python
# Extract features (done once)
with torch.no_grad():
    features = model.encode_image(images)
    
# Train linear classifier
classifier = nn.Linear(feature_dim, num_classes)
optimizer.zero_grad()
loss = criterion(classifier(features), labels)
loss.backward()
optimizer.step()
```

---

### 3. Full Fine-tuning

**What it does:**
- Unfreezes all or most of the CLIP model
- Trains end-to-end with backpropagation through entire network
- Uses lower learning rate for stability
- Implements gradient clipping and warmup

**When to use:**
- ✅ Maximum accuracy is required
- ✅ Sufficient GPU memory available
- ✅ Significant domain shift from pre-training
- ❌ Limited computational resources

**Pros:**
- Highest potential accuracy
- Full adaptation to target domain
- Can learn domain-specific features

**Cons:**
- Computationally expensive
- High GPU memory requirements
- Risk of overfitting on small datasets
- Longer training time

**Key Code Pattern:**
```python
# Unfreeze backbone
for param in model.visual.parameters():
    param.requires_grad = True
    
# Use different learning rates
optimizer = AdamW([
    {'params': model.visual.parameters(), 'lr': lr * 0.1},
    {'params': classifier.parameters(), 'lr': lr}
])

# Gradient clipping
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 4. LoRA (Low-Rank Adaptation)

**What it does:**
- Keeps original weights frozen
- Adds trainable low-rank matrices A and B to key layers
- ∆W = B @ A (where rank(A) = rank(B) = r << d)
- Can be merged with original weights after training

**When to use:**
- ✅ Need good accuracy with limited parameters
- ✅ Training multiple task-specific adapters
- ✅ Limited GPU memory
- ❌ When simplicity is more important than efficiency

**Pros:**
- Extremely parameter-efficient (~0.1% trainable)
- No inference latency (can merge adapters)
- Multiple adapters can be stored efficiently
- Good accuracy/efficiency trade-off

**Cons:**
- Slightly more complex implementation
- May not reach full fine-tuning accuracy
- Hyperparameters (rank, alpha) need tuning

**Key Code Pattern:**
```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        
    def forward(self, x):
        return x + (x @ self.lora_A @ self.lora_B) * self.scaling
```

---

### 5. BitFit (Bias-only Fine-tuning)

**What it does:**
- Freezes all weight parameters
- Unfreezes only bias terms
- Trains ~0.01% of total parameters

**When to use:**
- ✅ Extremely limited resources
- ✅ Very fast experimentation needed
- ✅ Good pre-trained model available
- ❌ Significant domain shift

**Pros:**
- Minimal parameters (~0.01%)
- Very fast training
- Extremely low memory usage
- Surprisingly effective

**Cons:**
- Lower accuracy than other methods
- Limited adaptation capability
- May not work well with large domain shifts

**Key Code Pattern:**
```python
# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False
    
# Unfreeze only bias terms
for name, param in model.named_parameters():
    if 'bias' in name:
        param.requires_grad = True
```

---

### 6. Prefix-tuning

**What it does:**
- Keeps backbone frozen
- Adds learnable prefix tokens to the input
- Trains prefix encoder (MLP) to generate continuous prompts
- Prepends prefix to all input sequences

**When to use:**
- ✅ Want input-level adaptation
- ✅ Need modular, composable adapters
- ✅ Limited parameters but better than BitFit
- ❌ Need maximum accuracy

**Pros:**
- Parameter-efficient (~0.5%)
- Modular and composable
- Can be applied to different inputs
- Good balance of efficiency and accuracy

**Cons:**
- Increases sequence length (slower inference)
- May not adapt as well as LoRA
- Hyperparameter sensitive (prefix length)

**Key Code Pattern:**
```python
class PrefixEncoder(nn.Module):
    def __init__(self, prefix_length, hidden_dim):
        super().__init__()
        self.prefix_tokens = nn.Parameter(torch.randn(prefix_length, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x):
        prefix = self.mlp(self.prefix_tokens)
        return torch.cat([prefix.unsqueeze(0).expand(x.size(0), -1, -1), x], dim=1)
```

## Performance vs. Efficiency Trade-off

```
High Accuracy
    ↑
    |                        [Full Fine-tune]
    |                              
    |                    [LoRA]
    |                         
    |              [Prefix-tuning]
    |         [Linear Probe]
    |              
    |    [BitFit]
    |         
    |  [Zero-Shot]
    |
    └──────────────────────────────────────→ Low Computational Cost
```

## Choosing the Right Method

### Decision Tree

1. **Do you have any training data?**
   - No → **Zero-Shot**
   - Yes → Continue

2. **Do you have GPU memory constraints?**
   - Severe (< 8GB) → **BitFit** or **Linear Probe**
   - Moderate (8-16GB) → **LoRA** or **Prefix-tuning**
   - None (> 16GB) → **Full Fine-tuning**

3. **How much accuracy do you need?**
   - Baseline → **Linear Probe**
   - Good → **LoRA** or **Prefix-tuning**
   - Best → **Full Fine-tuning**

4. **Do you need to train multiple task-specific models?**
   - Yes → **LoRA** (can store many adapters efficiently)
   - No → Choose based on resources

## Hyperparameter Recommendations

### Linear Probing
```python
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10
```

### Full Fine-tuning
```python
BACKBONE_LR = 1e-5  # 10x smaller than classifier
CLASSIFIER_LR = 1e-4
WARMUP_EPOCHS = 5
GRADIENT_CLIP = 1.0
NUM_EPOCHS = 30
```

### LoRA
```python
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
```

### BitFit
```python
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
```

### Prefix-tuning
```python
PREFIX_LENGTH = 10
PREFIX_HIDDEN_DIM = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
```

## Common Pitfalls and Solutions

### Overfitting
- **Problem**: Training accuracy much higher than validation
- **Solution**: 
  - Use more aggressive data augmentation
  - Reduce model complexity (use LoRA instead of full fine-tuning)
  - Increase weight decay
  - Use early stopping

### Underfitting
- **Problem**: Both training and validation accuracy are low
- **Solution**:
  - Increase model capacity (full fine-tuning instead of BitFit)
  - Train for more epochs
  - Increase learning rate
  - Reduce regularization

### GPU Out of Memory
- **Problem**: CUDA OOM errors during training
- **Solution**:
  - Reduce batch size
  - Use gradient accumulation
  - Switch to more efficient method (LoRA/BitFit)
  - Use mixed precision training (FP16)

### Slow Training
- **Problem**: Training takes too long
- **Solution**:
  - Increase batch size (if memory allows)
  - Use DataLoader with multiple workers
  - Enable CUDA benchmarking
  - Use mixed precision training

## Combining Methods

### Ensemble
Train multiple methods and ensemble predictions:
```python
# Average predictions from different methods
final_pred = (zero_shot_pred + linear_probe_pred + lora_pred) / 3
```

### Two-stage Training
1. Start with linear probing for quick baseline
2. Then apply LoRA or full fine-tuning for improvement

### Adapter Fusion
Train multiple adapters (LoRA, Prefix) and combine them for better performance.

## Further Reading

- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **BitFit**: [BitFit: Simple Parameter-efficient Fine-tuning](https://arxiv.org/abs/2106.10199)
- **Prefix-tuning**: [Prefix-Tuning: Optimizing Continuous Prompts](https://arxiv.org/abs/2101.00190)
- **CLIP**: [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)

## Running All Methods

To compare all methods in a single run:

```bash
python main.py --all
```

This will:
1. Run zero-shot evaluation
2. Train linear probe
3. Train with BitFit
4. Train with Prefix-tuning
5. Train with LoRA
6. Train with full fine-tuning
7. Generate comprehensive comparison table
8. Save all results and visualizations

Expected total runtime: 2-3 hours on a single GPU
