"""
Test script to verify dataset selection functionality
"""

import config

def test_dataset_config():
    """Test dataset configuration"""
    print("\n" + "="*70)
    print("TESTING DATASET CONFIGURATION")
    print("="*70)
    
    print(f"\nTotal datasets available: {len(config.AVAILABLE_DATASETS)}")
    
    for key, dataset in config.AVAILABLE_DATASETS.items():
        print(f"\n{key}. {dataset['display_name']}")
        print(f"   Name: {dataset['name']}")
        print(f"   Path: {dataset['path']}")
        print(f"   URL: {dataset['url']}")
    
    print("\n" + "="*70)
    print("Configuration test completed!")
    print("="*70)

if __name__ == "__main__":
    test_dataset_config()
