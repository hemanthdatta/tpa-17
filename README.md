# Foundation Model Fine-tuning on PlantVillage Dataset

**Project Focus**: Efficient fine-tuning of foundation models (CLIP, DINOv2, ViT) for domain-specific image classification in low-data regimes.

## Overview

This project investigates how to efficiently fine-tune foundation models to perform high-accuracy classification in specialized domains with limited labeled data. We use the **PlantVillage Crop Disease Dataset** to evaluate various fine-tuning strategies, balancing performance with computational and parameter efficiency.

### Problem Statement

Foundation models like CLIP excel at general-purpose tasks but often struggle with specialized domains due to domain shift. This project evaluates and compares fine-tuning strategies to achieve optimal performance in a **low-data regime** (limited labeled samples per class).

### Key Features

- **Domain-Specific Task**: Plant disease classification (38 classes)
- **Low-Data Regime**: 50 samples per class (~1,900 training images)
- **Multiple Fine-tuning Strategies**: Zero-shot, linear probing, full fine-tuning, and adapter methods
- **Parameter Efficiency Analysis**: Compare trainable parameters vs. performance
- **Comprehensive Evaluation**: Metrics, visualizations, and comparative analysis

## Implemented Methods

✅ **1. Zero-Shot Evaluation** - Testing foundation model without training  
✅ **2. Linear Probing** - Training linear classifier on frozen features  
✅ **3. Full Fine-tuning** - End-to-end training of entire model  
✅ **4. LoRA** - Low-Rank Adaptation (parameter-efficient)  
✅ **5. BitFit** - Bias-only fine-tuning  
✅ **6. Prefix-tuning** - Learning continuous prompts

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
├── README.md                 # Project overview & usage guide
├── DATASET_SETUP.md          # PlantVillage dataset setup guide
├── PROJECT_SUMMARY.md        # Technical documentation
├── GETTING_STARTED.md        # Step-by-step guide
└── requirements.txt          # Dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

**See [DATASET_SETUP.md](DATASET_SETUP.md) for detailed instructions.**

Quick steps:
1. Download PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Extract to: `./data/plantvillage/PlantVillage/`
3. Verify structure: `PlantVillage/class_name/*.jpg`

### 3. Run Experiments

```bash
# Run all experiments
python main.py --all

# Or run specific methods
python main.py --zero-shot --linear-probe
```

## Dataset: PlantVillage

### Why PlantVillage?

1. **Domain Shift**: Agricultural disease detection is distinct from web-scale pre-training data
2. **Real-World Application**: Practical impact in agriculture and food security
3. **Complexity**: 38 disease classes across multiple crops
4. **Low-Data Scenario**: Simulates realistic expert-labeled data scarcity

### Dataset Statistics

- **Classes**: 38 (plant diseases + healthy plants)
- **Low-Data Regime**: 50 samples per class
- **Training Set**: ~1,900 images (with limitation)
- **Validation Set**: ~11,400 images (15% of full data)
- **Test Set**: ~11,400 images (15% of full data)
- **Total Full Dataset**: ~54,000 images

### Classes Include

- Apple (scab, black rot, cedar rust, healthy)
- Corn (leaf spot, rust, blight, healthy)
- Grape (black rot, Esca, leaf blight, healthy)
- Potato (early blight, late blight, healthy)
- Tomato (10 different diseases/conditions)
- And more...

**Full class list**: See `config.py` or [DATASET_SETUP.md](DATASET_SETUP.md)

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

### Low-Data Regime Settings

Edit `config.py` to adjust the data regime:

```python
# Low-data regime configuration
SAMPLES_PER_CLASS = 50  # Number of training samples per class
USE_LIMITED_DATA = True  # Enable limited data mode
```

**Recommended settings**:
- **Extreme low-data**: 20 samples/class (~760 training images)
- **Standard low-data**: 50 samples/class (~1,900 training images)  ← **Current**
- **Moderate low-data**: 100 samples/class (~3,800 training images)

### Other Hyperparameters

```python
# Model Configuration
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Training Configuration
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping
BATCH_SIZE = 32
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

Zero-shot classification evaluates the foundation model without any training on the target dataset. It works by:

1. Creating text embeddings for each plant disease class using agriculture-specific templates
2. Computing image embeddings for each test image
3. Classifying by finding the text embedding most similar to each image embedding

**Key features:**
- Multiple text templates optimized for disease classification
- Normalized embeddings for cosine similarity
- No training required
- Tests domain transfer capability

**Expected accuracy: 40-60%** (lower due to domain shift from general to agricultural domain)

### 2. Linear Probing

Linear probing trains a simple linear classifier on top of frozen foundation model features:

1. Extract image features using the frozen CLIP/ViT model
2. Train a linear layer to map features to disease class labels
3. Use Adam optimizer with weight decay for regularization
4. Implement early stopping based on validation performance

**Key features:**
- Frozen backbone (no fine-tuning of foundation model)
- Efficient training on extracted features
- Works well in low-data regimes
- Model checkpointing

**Expected accuracy: 75-85%**

### 3. Full Fine-tuning

Full fine-tuning unfreezes all or part of the foundation model and trains end-to-end:

1. Unfreeze image encoder parameters
2. Add classification head for disease prediction
3. Train with lower learning rate to preserve pre-trained knowledge
4. Use gradient clipping for stability

**Key features:**
- Optional partial unfreezing (backbone vs. all layers)
- Separate learning rates for backbone and classifier
- Warmup schedule
- Gradient clipping
- Higher computational cost

**Expected accuracy: 85-92%**

### 4. LoRA (Low-Rank Adaptation)

LoRA adds trainable low-rank matrices to frozen weights, enabling parameter-efficient fine-tuning:

1. Inject LoRA layers into attention modules
2. Train only LoRA parameters (~0.1-1% of total)
3. Can merge adapters after training for no inference overhead

**Key features:**
- Extremely parameter-efficient (~0.1-1% trainable)
- No inference latency (can be merged)
- Configurable rank and alpha
- **Ideal for low-data regimes**

**Expected accuracy: 80-88%**

### 5. BitFit (Bias-only Fine-tuning)

BitFit trains only the bias terms while keeping all weights frozen:

1. Freeze all weight parameters
2. Unfreeze only bias terms (~0.01% of parameters)
3. Train with standard optimizer

**Key features:**
- Minimal parameters (~0.01% trainable)
- Very fast training
- Surprisingly effective for domain adaptation
- Minimal risk of overfitting

**Expected accuracy: 70-80%**

### 6. Prefix-tuning

Prefix-tuning learns continuous prompts prepended to the input:

1. Add learnable prefix tokens
2. Train prefix encoder MLP
3. Keep backbone frozen

**Key features:**
- Input-level adaptation
- Continuous prompt learning
- Modular and composable
- ~0.1-1% trainable parameters

**Expected accuracy: 75-83%**

## Expected Results (Low-Data Regime)

### Performance vs. Efficiency Trade-offs

| Method | Test Acc | Trainable % | Training Time | Memory | Best Use Case |
|--------|----------|-------------|---------------|--------|---------------|
| Zero-Shot | 40-60% | 0% | N/A | Low | Baseline, no labels |
| BitFit | 70-80% | 0.01% | Fast | Low | Quick adaptation |
| Prefix-tuning | 75-83% | ~0.5% | Fast | Low | Prompt learning |
| Linear Probe | 75-85% | <1% | Fast | Low | Feature extraction |
| LoRA | 80-88% | ~0.1% | Medium | Medium | **Best efficiency/performance** |
| Full Fine-tune | 85-92% | 100% | Slow | High | Maximum performance |

**Key Insights**:
- LoRA offers the best balance of performance and efficiency
- Linear probing is surprisingly effective for low-data scenarios
- Zero-shot shows the domain gap between pre-training and agriculture
- Full fine-tuning risks overfitting with limited data

*Note: Results may vary based on hyperparameters, data splits, and random seeds.*

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

### Dataset Not Found
```
FileNotFoundError: PlantVillage dataset not found
```
**Solution**: See [DATASET_SETUP.md](DATASET_SETUP.md) for download instructions.

### CUDA Out of Memory
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Use a smaller model: `--model ViT-B-32`
- Reduce number of workers in `config.py`: `NUM_WORKERS = 2`

### Slow Training
- Increase number of workers: Modify `NUM_WORKERS` in `config.py`
- Ensure GPU is available and being used
- Check CUDA installation: `torch.cuda.is_available()`

### Low Accuracy
- Try different learning rates: `--lr 0.0001` or `--lr 0.01`
- Increase training epochs: `--epochs 100`
- Experiment with different models: `--model ViT-L-14`
- Increase samples per class in `config.py`: `SAMPLES_PER_CLASS = 100`

### Class Imbalance Issues
The low-data regime ensures balanced sampling (same samples per class), but if you modify the code:
- Use weighted loss functions
- Apply class-balanced sampling
- Adjust classification thresholds

## Project Goals & Research Questions

This project addresses key questions in foundation model fine-tuning:

1. **How well do foundation models transfer to specialized domains?**
   - Measured via zero-shot performance
   - Expected domain gap due to agriculture vs. web-scale data

2. **What is the optimal fine-tuning strategy for low-data regimes?**
   - Compare parameter-efficient methods (LoRA, BitFit, Prefix)
   - vs. traditional approaches (linear probing, full fine-tuning)

3. **What are the performance-efficiency trade-offs?**
   - Accuracy vs. trainable parameters
   - Training time vs. final performance
   - Memory requirements vs. model capacity

4. **Can adapter methods match full fine-tuning with <1% parameters?**
   - Critical for resource-constrained deployments
   - Enables multi-task learning with shared backbone

## Extensions & Future Work

### Currently Implemented ✅

1. ✅ Zero-shot evaluation
2. ✅ Linear probing
3. ✅ Full fine-tuning
4. ✅ LoRA adaptation
5. ✅ BitFit (bias-only)
6. ✅ Prefix-tuning

### Potential Extensions

1. **Additional Foundation Models**:
   - DINOv2 (self-supervised)
   - SAM (Segment Anything Model)
   - Vision Transformers (supervised pre-training)

2. **Advanced Adapter Methods**:
   - Adapter fusion (combine multiple adapters)
   - (IA)³ (Infused Adapter by Inhibiting and Amplifying)
   - Compacter (combining adapters with LoRA)

3. **Data Efficiency**:
   - Active learning for sample selection
   - Semi-supervised learning with unlabeled data
   - Data augmentation strategies (MixUp, CutMix)

4. **Multi-Task Learning**:
   - Joint disease detection + severity estimation
   - Cross-crop transfer learning
   - Few-shot learning scenarios

5. **Practical Deployment**:
   - Model quantization (INT8, FP16)
   - Mobile deployment (TensorFlow Lite, ONNX)
   - Real-time inference optimization

## References

### Papers
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [BitFit: Simple Parameter-efficient Fine-tuning](https://arxiv.org/abs/2106.10199)
- [Prefix-Tuning: Optimizing Continuous Prompts](https://arxiv.org/abs/2101.00190)

### Datasets & Models
- [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

### Related Work
- Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision
- Foundation Models: Opportunities and Risks (Stanford HAI, 2021)

## Citation

If you use this code for your research, please cite:

```bibtex
@misc{plantvillage-finetuning-2025,
  title={Efficient Fine-tuning of Foundation Models for Plant Disease Classification},
  author={Your Name},
  year={2025},
  note={Low-data regime evaluation on PlantVillage dataset}
}
```

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue on GitHub.

---

**Happy fine-tuning! 🌱🔬**
