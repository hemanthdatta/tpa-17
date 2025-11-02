"""
Test script for Aircraft dataset configuration
Verifies CSV loading, class mapping, and data loader functionality
"""

import os
import config
from data_loader import get_csv_dataloaders, CSVDataset

def test_aircraft_configuration():
    """Test aircraft dataset configuration"""
    print("\n" + "="*70)
    print("AIRCRAFT DATASET CONFIGURATION TEST")
    print("="*70)
    
    # Set aircraft as current dataset
    config.CURRENT_DATASET = "6"
    config.DATASET_NAME = "aircraft"
    config.CURRENT_DATASET_PATH = "aircraft"
    
    # Get dataset info
    dataset_info = config.AVAILABLE_DATASETS["6"]
    
    print("\nDataset Information:")
    print(f"  Name: {dataset_info['display_name']}")
    print(f"  Path: {dataset_info['path']}")
    print(f"  Classes: {dataset_info['classes']}")
    print(f"  Training images: {dataset_info['num_train']}")
    print(f"  Validation images: {dataset_info['num_val']}")
    print(f"  Test images: {dataset_info['num_test']}")
    print(f"  Total images: {dataset_info['total_images']}")
    print(f"  Uses CSV: {dataset_info['use_csv']}")
    
    print("\nHierarchy Information:")
    hierarchy = dataset_info['hierarchy']
    print(f"  Levels: {', '.join(hierarchy['levels'])}")
    print(f"  Variants (used): {hierarchy['num_variants']}")
    print(f"  Families: {hierarchy['num_families']}")
    print(f"  Manufacturers: {hierarchy['num_manufacturers']}")
    print(f"  Classification level: {hierarchy['classification_level']}")
    
    print("\nCSV Files:")
    for split, filename in dataset_info['csv_files'].items():
        print(f"  {split}: {filename}")
    
    # Check if dataset exists
    dataset_path = os.path.join(config.DATA_ROOT, dataset_info['path'])
    print(f"\nDataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("\n❌ Dataset directory not found!")
        print(f"   Expected: {dataset_path}")
        print("\nPlease download and extract the aircraft dataset to continue.")
        return False
    else:
        print("✓ Dataset directory found")
    
    # Check CSV files
    csv_files_exist = []
    for split, filename in dataset_info['csv_files'].items():
        csv_path = os.path.join(dataset_path, filename)
        exists = os.path.exists(csv_path)
        csv_files_exist.append(exists)
        status = "✓" if exists else "❌"
        print(f"{status} {split}.csv: {csv_path}")
    
    if not all(csv_files_exist):
        print("\n❌ Some CSV files are missing!")
        print("Please ensure all CSV files (train.csv, val.csv, test.csv) are in the dataset directory.")
        return False
    
    # Try to load a sample dataset
    print("\n" + "="*70)
    print("TESTING CSV DATASET LOADING")
    print("="*70)
    
    try:
        train_csv = os.path.join(dataset_path, dataset_info['csv_files']['train'])
        print(f"\nLoading training CSV: {train_csv}")
        
        sample_dataset = CSVDataset(
            csv_file=train_csv,
            root_dir=dataset_path,
            transform=None
        )
        
        print(f"✓ CSV loaded successfully")
        print(f"  Total samples: {len(sample_dataset)}")
        print(f"  Number of classes: {len(sample_dataset.classes)}")
        print(f"  First 10 classes: {sample_dataset.classes[:10]}")
        
        # Try to load a sample image
        print("\nTesting image loading...")
        try:
            image, label = sample_dataset[0]
            print(f"✓ Image loaded successfully")
            print(f"  Image size: {image.size}")
            print(f"  Label: {label}")
            print(f"  Class name: {sample_dataset.idx_to_class[label]}")
        except Exception as e:
            print(f"❌ Error loading sample image: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Error loading CSV dataset: {e}")
        return False
    
    # Try to load full dataloaders
    print("\n" + "="*70)
    print("TESTING FULL DATA LOADER")
    print("="*70)
    
    try:
        # Temporarily disable limited data for testing
        original_limited = config.USE_LIMITED_DATA
        config.USE_LIMITED_DATA = False
        
        train_loader, val_loader, test_loader = get_csv_dataloaders(
            batch_size=16,
            use_clip_transforms=False
        )
        
        config.USE_LIMITED_DATA = original_limited
        
        print(f"✓ Data loaders created successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Get a batch
        print("\nTesting batch loading...")
        images, labels = next(iter(train_loader))
        print(f"✓ Batch loaded successfully")
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        print(f"  Number of unique labels in batch: {len(labels.unique())}")
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nThe aircraft dataset is properly configured and ready to use.")
    print("\nTo run experiments:")
    print("  python main.py --dataset 6 --all")
    print("  python main.py --dataset 6 --zero-shot")
    print("  python main.py --dataset 6 --linear-probe")
    
    return True


if __name__ == "__main__":
    success = test_aircraft_configuration()
    exit(0 if success else 1)
