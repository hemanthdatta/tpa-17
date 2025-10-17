# CLIP on Flowers102 Dataset

This project implements **Zero-Shot Evaluation** and **Linear Probing** using OpenCLIP on the Oxford Flowers102 dataset.

## Features

- **Zero-Shot Classification**: Evaluate CLIP's ability to classify flowers without any training
- **Linear Probing**: Train a linear classifier on top of frozen CLIP features
- Comprehensive evaluation metrics and visualizations
- Support for different CLIP model architectures
- Easy-to-use command-line interface

## Project Structure

```
tpa-17/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Dataset loading and preprocessing
├── zero_shot.py          # Zero-shot classification implementation
├── linear_probe.py       # Linear probing implementation
├── utils.py              # Utility functions for visualization and metrics
├── main.py               # Main script to run experiments
├── requirements.txt      # Python dependencies
└── README.md            # This file
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

**Zero-Shot Evaluation Only:**
```bash
python main.py --zero-shot
```

**Linear Probing Only:**
```bash
python main.py --linear-probe
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

## Expected Results

Typical accuracy ranges on Flowers102:

- **Zero-Shot**: 60-75% (depending on model size)
- **Linear Probing**: 85-95% (depending on model size)

## Model Options

You can experiment with different CLIP models:

| Model | Parameters | Description |
|-------|------------|-------------|
| ViT-B-32 | 151M | Base model, good balance |
| ViT-B-16 | 149M | Base with smaller patches |
| ViT-L-14 | 428M | Large model, better accuracy |
| RN50 | 102M | ResNet-50 backbone |

## Extending the Project

To implement additional evaluation methods (as mentioned in TPA17):

1. **Fine-tuning**: Modify `linear_probe.py` to unfreeze CLIP layers
2. **Few-shot Learning**: Implement k-shot evaluation with limited samples
3. **Adapter Layers**: Add trainable adapter modules
4. **Prompt Tuning**: Learn continuous prompts instead of discrete text

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
