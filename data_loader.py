"""
Data loader for multiple datasets with low-data regime support
Supports both ImageFolder and CSV-based dataset formats
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import numpy as np
from collections import defaultdict
from PIL import Image
import config


class CSVDataset(Dataset):
    """
    Custom Dataset for CSV-based datasets (like Aircraft)
    
    CSV Format:
        - filename: image file name
        - Classes: class name (e.g., '707-320', 'A318')
        - Labels: categorical value (0-99 for 100 classes)
    """
    def __init__(self, csv_file: str, root_dir: str, transform=None, class_to_idx=None):
        """
        Args:
            csv_file: Path to CSV file
            root_dir: Root directory with images
            transform: Transforms to apply to images
            class_to_idx: Optional dictionary mapping class names to indices
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Get unique classes and create mapping if not provided
        if class_to_idx is None:
            unique_classes = sorted(self.data_frame['Classes'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.classes = unique_classes
        else:
            self.class_to_idx = class_to_idx
            self.classes = [cls for cls, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image filename and label
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['filename'])
        
        # Use 'Labels' column if available, otherwise map from 'Classes'
        if 'Labels' in self.data_frame.columns:
            label = int(self.data_frame.iloc[idx]['Labels'])
        else:
            class_name = self.data_frame.iloc[idx]['Classes']
            label = self.class_to_idx[class_name]
        
        # Load image
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_current_dataset_path():
    """
    Get the current dataset path from config
    
    Returns:
        Dataset path string
    """
    if config.CURRENT_DATASET_PATH:
        return config.CURRENT_DATASET_PATH
    else:
        # Default to plantvillage for backward compatibility
        return "plantvillage/PlantVillage"


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


def get_csv_dataloaders(
    root: str = config.DATA_ROOT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_clip_transforms: bool = False,
    clip_preprocess = None,
    use_limited_data: bool = config.USE_LIMITED_DATA,
    samples_per_class: int = config.SAMPLES_PER_CLASS,
    dataset_path: str = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders for CSV-based datasets (like Aircraft)
    
    Args:
        root: Root directory for dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        use_clip_transforms: Whether to use CLIP's preprocessing transforms
        clip_preprocess: CLIP preprocessing function (required if use_clip_transforms=True)
        use_limited_data: Whether to use limited data (low-data regime)
        samples_per_class: Number of samples per class in limited mode
        dataset_path: Custom dataset path (overrides default)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create data directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    # Get dataset configuration
    dataset_info = config.AVAILABLE_DATASETS[config.CURRENT_DATASET]
    csv_files = dataset_info['csv_files']
    
    # Set up transforms
    if use_clip_transforms:
        if clip_preprocess is None:
            raise ValueError("clip_preprocess must be provided when use_clip_transforms=True")
        train_transform = clip_preprocess
        test_transform = clip_preprocess
    else:
        train_transform = get_plantvillage_transforms(is_train=True)
        test_transform = get_plantvillage_transforms(is_train=False)
    
    # Set up dataset path
    if dataset_path:
        full_dataset_path = os.path.join(root, dataset_path)
    else:
        full_dataset_path = os.path.join(root, get_current_dataset_path())
    
    # Check if dataset directory exists
    if not os.path.exists(full_dataset_path):
        print(f"\n{'='*70}")
        print("DATASET NOT FOUND")
        print(f"{'='*70}")
        print(f"Expected location: {full_dataset_path}")
        print(f"\nDataset URL: {dataset_info['url']}")
        print("\nFor CSV-based datasets, ensure:")
        print("  1. Images are in the dataset folder")
        print("  2. CSV files (train.csv, val.csv, test.csv) are present")
        print("  3. CSV has columns: filename, Classes, Labels")
        print(f"{'='*70}\n")
        raise FileNotFoundError(f"Dataset not found at {full_dataset_path}")
    
    # Check for CSV files
    train_csv = os.path.join(full_dataset_path, csv_files['train'])
    val_csv = os.path.join(full_dataset_path, csv_files['val'])
    test_csv = os.path.join(full_dataset_path, csv_files['test'])
    
    if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
        print(f"\n{'='*70}")
        print("CSV FILES NOT FOUND")
        print(f"{'='*70}")
        print(f"Expected files:")
        print(f"  - {train_csv}")
        print(f"  - {val_csv}")
        print(f"  - {test_csv}")
        print(f"{'='*70}\n")
        raise FileNotFoundError("Required CSV files not found")
    
    # Load train dataset to get class mapping
    train_dataset_full = CSVDataset(
        csv_file=train_csv,
        root_dir=full_dataset_path,
        transform=train_transform
    )
    
    # Use the same class mapping for all splits
    class_to_idx = train_dataset_full.class_to_idx
    classes = train_dataset_full.classes
    
    # Update config with class names
    config.PLANT_DISEASE_CLASSES = classes
    print(f"\nDetected {len(classes)} classes in the dataset")
    
    # Apply limited data regime to training set if enabled
    if use_limited_data:
        print(f"\n{'='*70}")
        print(f"LOW-DATA REGIME ENABLED: {samples_per_class} samples per class")
        print(f"{'='*70}")
        
        # Create a temporary dataset without transforms for limiting
        temp_train = CSVDataset(
            csv_file=train_csv,
            root_dir=full_dataset_path,
            transform=None,
            class_to_idx=class_to_idx
        )
        
        # Create limited training set
        limited_train_subset = create_limited_dataset(
            temp_train,
            samples_per_class=samples_per_class
        )
        
        # Get the actual indices
        limited_indices = limited_train_subset.indices
        
        # Create final training dataset with transforms
        train_dataset = CSVDataset(
            csv_file=train_csv,
            root_dir=full_dataset_path,
            transform=train_transform,
            class_to_idx=class_to_idx
        )
        train_dataset = Subset(train_dataset, limited_indices)
    else:
        train_dataset = train_dataset_full
    
    # Load validation and test datasets
    val_dataset = CSVDataset(
        csv_file=val_csv,
        root_dir=full_dataset_path,
        transform=test_transform,
        class_to_idx=class_to_idx
    )
    
    test_dataset = CSVDataset(
        csv_file=test_csv,
        root_dir=full_dataset_path,
        transform=test_transform,
        class_to_idx=class_to_idx
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
    
    print(f"\nDataset loaded successfully (CSV-based):")
    print(f"  Total classes: {len(classes)}")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    if use_limited_data:
        print(f"  Low-data regime: {samples_per_class} samples/class")
    
    return train_loader, val_loader, test_loader


def get_plantvillage_dataloaders(
    root: str = config.DATA_ROOT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    use_clip_transforms: bool = False,
    clip_preprocess = None,
    use_limited_data: bool = config.USE_LIMITED_DATA,
    samples_per_class: int = config.SAMPLES_PER_CLASS,
    dataset_path: str = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders for any dataset
    Supports both ImageFolder and CSV-based datasets
    
    Args:
        root: Root directory for dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        use_clip_transforms: Whether to use CLIP's preprocessing transforms
        clip_preprocess: CLIP preprocessing function (required if use_clip_transforms=True)
        use_limited_data: Whether to use limited data (low-data regime)
        samples_per_class: Number of samples per class in limited mode
        dataset_path: Custom dataset path (overrides default)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Check if current dataset uses CSV format
    dataset_info = None
    if config.CURRENT_DATASET and config.CURRENT_DATASET in config.AVAILABLE_DATASETS:
        dataset_info = config.AVAILABLE_DATASETS[config.CURRENT_DATASET]
        if dataset_info.get('use_csv', False):
            return get_csv_dataloaders(
                root=root,
                batch_size=batch_size,
                num_workers=num_workers,
                use_clip_transforms=use_clip_transforms,
                clip_preprocess=clip_preprocess,
                use_limited_data=use_limited_data,
                samples_per_class=samples_per_class,
                dataset_path=dataset_path
            )
    
    # Original ImageFolder-based loading
    # Create data directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    if use_clip_transforms:
        if clip_preprocess is None:
            raise ValueError("clip_preprocess must be provided when use_clip_transforms=True")
        train_transform = clip_preprocess
        test_transform = clip_preprocess
    else:
        train_transform = get_plantvillage_transforms(is_train=True)
        test_transform = get_plantvillage_transforms(is_train=False)
    
    # Load full dataset using ImageFolder
    # Use custom path if provided, otherwise use current dataset from config
    if dataset_path:
        full_dataset_path = os.path.join(root, dataset_path)
    else:
        full_dataset_path = os.path.join(root, get_current_dataset_path())
    
    # Check if dataset exists
    if not os.path.exists(full_dataset_path):
        print(f"\n{'='*70}")
        print("DATASET NOT FOUND")
        print(f"{'='*70}")
        print(f"Expected location: {full_dataset_path}")
        print("\nPlease download the dataset and extract to the correct location")
        print("Ensure structure: dataset_path/class_name/*.jpg")
        if config.CURRENT_DATASET and config.CURRENT_DATASET in config.AVAILABLE_DATASETS:
            dataset_info = config.AVAILABLE_DATASETS[config.CURRENT_DATASET]
            print(f"\nDataset URL: {dataset_info['url']}")
        print(f"{'='*70}\n")
        raise FileNotFoundError(f"Dataset not found at {full_dataset_path}")
    
    # Load the full dataset
    full_dataset = datasets.ImageFolder(
        root=full_dataset_path,
        transform=train_transform
    )
    
    # Update config with class names
    config.PLANT_DISEASE_CLASSES = full_dataset.classes
    print(f"\nDetected {len(full_dataset.classes)} classes in the dataset")
    
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

