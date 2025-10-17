"""
Quick start example for CLIP on Flowers102
This script provides a simple example of how to use the implemented modules
"""

import torch
import open_clip
from data_loader import get_flowers_dataloaders
import config


def quick_example():
    """
    Quick example showing how to use CLIP with the Flowers dataset
    """
    print("="*60)
    print("CLIP on Flowers102 - Quick Start Example")
    print("="*60)
    
    # 1. Load CLIP model
    print("\n1. Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL,
        pretrained=config.CLIP_PRETRAINED
    )
    model = model.to(config.DEVICE)
    model.eval()
    print(f"   Model loaded: {config.CLIP_MODEL}")
    print(f"   Device: {config.DEVICE}")
    
    # 2. Load dataset
    print("\n2. Loading Flowers102 dataset...")
    train_loader, val_loader, test_loader = get_flowers_dataloaders(
        use_clip_transforms=True,
        clip_preprocess=preprocess,
        batch_size=4  # Small batch for demo
    )
    
    # 3. Get a sample batch
    print("\n3. Getting a sample batch...")
    images, labels = next(iter(test_loader))
    print(f"   Batch shape: {images.shape}")
    print(f"   Labels: {labels}")
    
    # 4. Extract image features
    print("\n4. Extracting image features...")
    images = images.to(config.DEVICE)
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    print(f"   Image features shape: {image_features.shape}")
    
    # 5. Create text embeddings for a few classes
    print("\n5. Creating text embeddings for sample classes...")
    sample_classes = config.FLOWER_CLASSES[:5]  # First 5 classes
    print(f"   Sample classes: {sample_classes}")
    
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
    texts = [f"a photo of a {cls}" for cls in sample_classes]
    text_tokens = tokenizer(texts).to(config.DEVICE)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    print(f"   Text features shape: {text_features.shape}")
    
    # 6. Compute similarity
    print("\n6. Computing image-text similarity...")
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(f"   Similarity shape: {similarity.shape}")
    print(f"   Sample similarities:\n{similarity[0]}")
    
    predictions = similarity.argmax(dim=-1)
    print(f"   Predictions: {predictions}")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("\nTo run full experiments:")
    print("  - Zero-shot:     python main.py --zero-shot")
    print("  - Linear probe:  python main.py --linear-probe")
    print("  - All:           python main.py --all")
    print("="*60)


if __name__ == "__main__":
    quick_example()
