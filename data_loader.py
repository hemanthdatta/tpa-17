"""
Data loader for PlantVillage dataset with low-data regime support
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np
from collections import defaultdict
import config


def get_plantvillage_transforms(is_train: bool = True):
    """
    Get transforms for PlantVillage dataset
    
    Args:
        is_train: Whether to use training transforms (with augmentation)
    
    Returns:
        torchvision.transforms.Compose object
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_clip_transforms(preprocess):
    """
    Returns the preprocessing transform from CLIP model
    
    Args:
        preprocess: The preprocessing function from CLIP model
    
    Returns:
        The CLIP preprocessing transform
    """
    return preprocess


def create_limited_dataset(dataset, samples_per_class: int, seed: int = 42):
    """
    Create a limited dataset with specified number of samples per class
    For simulating low-data regime scenarios
    
    Args:
        dataset: The full dataset
        samples_per_class: Number of samples to keep per class
        seed: Random seed for reproducibility
    
    Returns:
        Subset of the dataset with limited samples
    """
    np.random.seed(seed)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)
    
    # Sample indices from each class
    selected_indices = []
    for class_label, indices in class_indices.items():
        # Shuffle and select samples_per_class samples
        np.random.shuffle(indices)
        selected_indices.extend(indices[:samples_per_class])
    
    print(f"  Limited dataset: {len(selected_indices)} samples ({samples_per_class} per class)")
    return Subset(dataset, selected_indices)


def get_plantvillage_dataloaders(
    root: str = config.DATA_ROOT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_clip_transforms: bool = False,
    clip_preprocess = None,
    use_limited_data: bool = config.USE_LIMITED_DATA,
    samples_per_class: int = config.SAMPLES_PER_CLASS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders for PlantVillage dataset
    
    Args:
        root: Root directory for dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        use_clip_transforms: Whether to use CLIP's preprocessing transforms
        clip_preprocess: CLIP preprocessing function (required if use_clip_transforms=True)
        use_limited_data: Whether to use limited data (low-data regime)
        samples_per_class: Number of samples per class in limited mode
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create data directory if it doesn't exist
    data_path = os.path.join(root, "plantvillage")
    os.makedirs(data_path, exist_ok=True)
    
    if use_clip_transforms:
        if clip_preprocess is None:
            raise ValueError("clip_preprocess must be provided when use_clip_transforms=True")
        train_transform = clip_preprocess
        test_transform = clip_preprocess
    else:
        train_transform = get_plantvillage_transforms(is_train=True)
        test_transform = get_plantvillage_transforms(is_train=False)
    
    # Load full dataset using ImageFolder
    # PlantVillage dataset should be organized in class folders
    full_dataset_path = os.path.join(data_path, "PlantVillage")
    
    # Check if dataset exists
    if not os.path.exists(full_dataset_path):
        print(f"\n{'='*70}")
        print("PLANTVILLAGE DATASET NOT FOUND")
        print(f"{'='*70}")
        print(f"Expected location: {full_dataset_path}")
        print("\nPlease download the PlantVillage dataset:")
        print("1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("2. Download and extract to: {data_path}")
        print("3. Ensure structure: PlantVillage/class_name/*.jpg")
        print(f"{'='*70}\n")
        raise FileNotFoundError(f"PlantVillage dataset not found at {full_dataset_path}")
    
    # Load the full dataset
    full_dataset = datasets.ImageFolder(
        root=full_dataset_path,
        transform=train_transform
    )
    
    # Create train/val/test splits (70/15/15)
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_split = int(0.7 * num_samples)
    val_split = int(0.85 * num_samples)
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    # Create datasets for each split
    train_dataset_full = Subset(full_dataset, train_indices)
    
    # Apply limited data regime to training set if enabled
    if use_limited_data:
        print(f"\n{'='*70}")
        print(f"LOW-DATA REGIME ENABLED: {samples_per_class} samples per class")
        print(f"{'='*70}")
        
        # Create a temporary dataset with proper transform for limiting
        temp_dataset = datasets.ImageFolder(
            root=full_dataset_path,
            transform=None  # No transform for indexing
        )
        temp_train = Subset(temp_dataset, train_indices)
        
        # Create limited training set
        limited_train_subset = create_limited_dataset(
            temp_train,
            samples_per_class=samples_per_class
        )
        
        # Get the actual indices from the limited subset
        limited_indices = [train_indices[i] for i in limited_train_subset.indices]
        
        # Create final training dataset with transforms
        train_dataset = Subset(full_dataset, limited_indices)
    else:
        train_dataset = train_dataset_full
    
    # Val and test datasets (no limitation, use test_transform)
    val_dataset_temp = datasets.ImageFolder(root=full_dataset_path, transform=test_transform)
    test_dataset_temp = datasets.ImageFolder(root=full_dataset_path, transform=test_transform)
    
    val_dataset = Subset(val_dataset_temp, val_indices)
    test_dataset = Subset(test_dataset_temp, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataset loaded successfully:")
    print(f"  Total classes: {len(full_dataset.classes)}")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    if use_limited_data:
        print(f"  Low-data regime: {samples_per_class} samples/class")
    
    return train_loader, val_loader, test_loader


# Backward compatibility alias
get_flowers_dataloaders = get_plantvillage_dataloaders


if __name__ == "__main__":
    # Test the data loader
    print("\nTesting PlantVillage data loader...")
    train_loader, val_loader, test_loader = get_plantvillage_dataloaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")

