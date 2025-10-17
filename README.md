# CLIP on Flowers102 Dataset

This project implements comprehensive evaluation and fine-tuning methods using OpenCLIP on the Oxford Flowers102 dataset.

## Implemented Methods

✅ **1. Zero-Shot Evaluation** - Testing CLIP without training  
✅ **2. Linear Probing** - Training linear classifier on frozen features  
✅ **3. Full Fine-tuning** - End-to-end training of entire model  
✅ **4. LoRA** - Low-Rank Adaptation (parameter-efficient)  
✅ **5. BitFit** - Bias-only fine-tuning  
✅ **6. Prefix-tuning** - Learning continuous prompts  

## Features

- **Zero-Shot Classification**: Evaluate CLIP's ability to classify flowers without any training
- **Linear Probing**: Train a linear classifier on top of frozen CLIP features
- **Full Fine-tuning**: Unfreeze and train the entire CLIP model end-to-end
- **Parameter-Efficient Methods**: LoRA, BitFit, and Prefix-tuning adapters
- Comprehensive evaluation metrics and visualizations
- Support for different CLIP model architectures
- Easy-to-use command-line interface

## Project Structure

```
tpa-17/
├── config.py                  # Configuration & hyperparameters
├── data_loader.py            # Dataset loading & preprocessing
├── zero_shot.py              # Zero-shot classification
├── linear_probe.py           # Linear probing implementation
├── full_finetuning.py        # Full fine-tuning implementation
├── adapter_finetuning.py     # LoRA, BitFit, Prefix-tuning
├── utils.py                  # Visualization & metrics
├── main.py                   # Main experiment runner
├── requirements.txt          # Dependencies
├── README.md                 # User guide
├── PROJECT_SUMMARY.md        # Technical documentation
└── GETTING_STARTED.md        # Step-by-step guide
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

The Oxford Flowers102 dataset will be automatically downloaded when you run the experiments for the first time. It will be saved to the `./data` directory.

## Usage

### Run All Experiments (Recommended)

```bash
python main.py --all
```

### Run Individual Experiments

**Zero-Shot Evaluation:**
```bash
python main.py --zero-shot
```

**Linear Probing:**
```bash
python main.py --linear-probe
```

**Full Fine-tuning:**
```bash
python main.py --full-finetune
```

**LoRA Fine-tuning:**
```bash
python main.py --lora
```

**BitFit Fine-tuning:**
```bash
python main.py --bitfit
```

**Prefix-tuning:**
```bash
python main.py --prefix
```

**All Adapter Methods:**
```bash
python main.py --adapters
```

### Advanced Options

```bash
python main.py --all --model ViT-L-14 --pretrained laion2b_s32b_b82k --batch-size 64 --epochs 100 --lr 0.001
```

#### Available Arguments:

- `--zero-shot`: Run zero-shot evaluation
- `--linear-probe`: Run linear probing
- `--all`: Run all experiments
- `--model`: CLIP model architecture (default: ViT-B-32)
- `--pretrained`: Pretrained weights identifier (default: laion2b_s34b_b79k)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs for linear probing (default: 50)
- `--lr`: Learning rate for linear probing (default: 0.001)

## Configuration

You can modify hyperparameters in `config.py`:

```python
# Model Configuration
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Training Configuration
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping
```

## Output

Results are saved to the `./results` directory:

- **JSON files**: Raw results with predictions and labels
- **Confusion matrices**: Visual representation of model performance
- **Training curves**: Loss and accuracy over epochs (for linear probing)
- **Classification reports**: Precision, recall, F1-score per class
- **Comparison plots**: Zero-shot vs linear probing performance

## Methods Implemented

### 1. Zero-Shot Classification

Zero-shot classification evaluates CLIP without any training on the target dataset. It works by:

1. Creating text embeddings for each flower class using multiple templates
2. Computing image embeddings for each test image
3. Classifying by finding the text embedding most similar to each image embedding

**Key features:**
- Multiple text templates for better generalization
- Normalized embeddings for cosine similarity
- No training required

**Expected accuracy: 65-75%**

### 2. Linear Probing

Linear probing trains a simple linear classifier on top of frozen CLIP features:

1. Extract image features using the frozen CLIP model
2. Train a linear layer to map features to class labels
3. Use Adam optimizer with weight decay for regularization
4. Implement early stopping based on validation performance

**Key features:**
- Frozen CLIP backbone (no fine-tuning)
- Efficient training on extracted features
- Early stopping to prevent overfitting
- Model checkpointing

**Expected accuracy: 85-92%**

### 3. Full Fine-tuning

Full fine-tuning unfreezes all or part of the CLIP model and trains end-to-end:

1. Unfreeze image encoder parameters
2. Add classification head
3. Train with lower learning rate
4. Use gradient clipping for stability

**Key features:**
- Optional partial unfreezing (backbone vs. all layers)
- Separate learning rates for backbone and classifier
- Warmup schedule
- Gradient clipping

**Expected accuracy: 90-95%**

### 4. LoRA (Low-Rank Adaptation)

LoRA adds trainable low-rank matrices to frozen weights, enabling parameter-efficient fine-tuning:

1. Inject LoRA layers into attention modules
2. Train only LoRA parameters (~0.1% of total)
3. Merge adapters after training if needed

**Key features:**
- Extremely parameter-efficient (~0.1% trainable)
- No inference latency (can be merged)
- Configurable rank and alpha

**Expected accuracy: 88-93%**

### 5. BitFit (Bias-only Fine-tuning)

BitFit trains only the bias terms while keeping all weights frozen:

1. Freeze all weight parameters
2. Unfreeze only bias terms
3. Train with standard optimizer

**Key features:**
- Minimal parameters (~0.01% trainable)
- Very fast training
- Surprisingly effective

**Expected accuracy: 82-87%**

### 6. Prefix-tuning

Prefix-tuning learns continuous prompts prepended to the input:

1. Add learnable prefix tokens
2. Train prefix encoder MLP
3. Keep backbone frozen

**Key features:**
- Input-level adaptation
- Continuous prompt learning
- Modular and composable

**Expected accuracy: 85-90%**

## Expected Results

### Performance Comparison

| Method | Train | Val | Test | Trainable % |
|--------|-------|-----|------|-------------|
| Zero-Shot | 68% | 68% | 68% | 0% |
| Linear Probe | 92% | 88% | 88% | <1% |
| BitFit | 87% | 84% | 84% | 0.01% |
| Prefix-tuning | 90% | 87% | 87% | ~0.5% |
| LoRA | 94% | 90% | 90% | ~0.1% |
| Full Fine-tune | 98% | 92% | 92% | 100% |

*Note: Actual results may vary based on model size, hyperparameters, and random seeds.*

## Model Options

You can experiment with different CLIP models:

| Model | Parameters | Description |
|-------|------------|-------------|
| ViT-B-32 | 151M | Base model, good balance |
| ViT-B-16 | 149M | Base with smaller patches |
| ViT-L-14 | 428M | Large model, better accuracy |
| RN50 | 102M | ResNet-50 backbone |

## Extending the Project

### Currently Implemented ✅

1. **Zero-shot Classification** ✅
2. **Linear Probing** ✅
3. **Full Fine-tuning** ✅
4. **LoRA** ✅
5. **BitFit** ✅
6. **Prefix-tuning** ✅

### Future Extensions

Additional methods that could be added:

1. **Adapter Fusion**: Combine multiple trained adapters
2. **Mixed Precision Training**: Use FP16 or BF16 for faster training
3. **Knowledge Distillation**: Distill large model to smaller one
4. **Data Augmentation**: Advanced augmentation strategies
5. **Ensemble Methods**: Combine multiple fine-tuned models

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 16`
- Use a smaller model: `--model ViT-B-32`

### Slow Training
- Increase number of workers: Modify `NUM_WORKERS` in `config.py`
- Use GPU if available

### Dataset Download Issues
- Manually download from [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- Place in `./data/flowers-102/` directory

## References

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Oxford Flowers102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## License

This project is for educational purposes.
