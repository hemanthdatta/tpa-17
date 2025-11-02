"""
Quick script to display available datasets
Run this to see what datasets you can use
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

def show_datasets():
    """Display all available datasets with details"""
    print("\n" + "="*80)
    print(" " * 25 + "AVAILABLE DATASETS")
    print("="*80)
    
    for key, dataset in sorted(config.AVAILABLE_DATASETS.items()):
        print(f"\n📁 Option {key}: {dataset['display_name']}")
        print(f"   └─ Name: {dataset['name']}")
        print(f"   └─ Path: ./data/{dataset['path']}")
        print(f"   └─ Kaggle URL: {dataset['url']}")
        
        # Check if dataset exists
        dataset_path = os.path.join(config.DATA_ROOT, dataset['path'])
        if os.path.exists(dataset_path):
            print(f"   └─ Status: ✅ INSTALLED")
            try:
                # Try to count folders (classes)
                if os.path.isdir(dataset_path):
                    classes = [d for d in os.listdir(dataset_path) 
                              if os.path.isdir(os.path.join(dataset_path, d))]
                    if classes:
                        print(f"   └─ Classes: {len(classes)} detected")
            except:
                pass
        else:
            print(f"   └─ Status: ❌ NOT INSTALLED")
    
    print("\n" + "="*80)
    print("\nTo use a dataset:")
    print("  Interactive: python main.py --knn-zero-shot")
    print("  Command Line: python main.py --dataset <number> --knn-zero-shot")
    print("\nExample: python main.py --dataset 1 --linear-probe")
    print("="*80 + "\n")

if __name__ == "__main__":
    show_datasets()
