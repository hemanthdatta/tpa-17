"""
Data loader for Flowers102 dataset
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple
import config


def get_flowers_transforms(is_train: bool = True):
    """
    Get transforms for Flowers102 dataset
    
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


def get_flowers_dataloaders(
    root: str = config.DATA_ROOT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_clip_transforms: bool = False,
    clip_preprocess = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders for Flowers102 dataset
    
    Args:
        root: Root directory for dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        use_clip_transforms: Whether to use CLIP's preprocessing transforms
        clip_preprocess: CLIP preprocessing function (required if use_clip_transforms=True)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create data directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    if use_clip_transforms:
        if clip_preprocess is None:
            raise ValueError("clip_preprocess must be provided when use_clip_transforms=True")
        train_transform = clip_preprocess
        test_transform = clip_preprocess
    else:
        train_transform = get_flowers_transforms(is_train=True)
        test_transform = get_flowers_transforms(is_train=False)
    
    # Load datasets
    # Note: Flowers102 has predefined splits
    train_dataset = datasets.Flowers102(
        root=root,
        split='train',
        transform=train_transform,
        download=True
    )
    
    val_dataset = datasets.Flowers102(
        root=root,
        split='val',
        transform=test_transform,
        download=True
    )
    
    test_dataset = datasets.Flowers102(
        root=root,
        split='test',
        transform=test_transform,
        download=True
    )
    
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
    
    print(f"Dataset loaded successfully:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    train_loader, val_loader, test_loader = get_flowers_dataloaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: [{labels.min()}, {labels.max()}]")
