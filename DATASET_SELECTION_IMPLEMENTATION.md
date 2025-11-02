# Dataset Selection Feature - Implementation Summary

## Overview
Implemented a flexible dataset selection system that allows users to easily switch between multiple datasets when running experiments.

## Changes Made

### 1. Configuration Updates (`config.py`)

#### Added:
- `AVAILABLE_DATASETS` dictionary containing 5 dataset configurations:
  - PlantVillage Plant Disease
  - NEU Surface Defect Database
  - Goldenhar CFID
  - Multi-Class Semiconductor Wafer Image Dataset
  - PCB Defect (Modified)
- `CURRENT_DATASET` - Runtime variable to track selected dataset
- `CURRENT_DATASET_PATH` - Runtime variable for dataset path
- Modified `PLANT_DISEASE_CLASSES` to be auto-detected (empty list by default)
- Updated `TEXT_TEMPLATES` to be more generic for all datasets

#### Dataset Configuration Structure:
```python
"1": {
    "name": "plantvillage",
    "display_name": "PlantVillage Plant Disease",
    "path": "plantvillage/PlantVillage",
    "url": "https://www.kaggle.com/datasets/...",
    "classes": None  # Auto-detected
}
```

### 2. Data Loader Updates (`data_loader.py`)

#### Added:
- `get_current_dataset_path()` - Helper function to retrieve current dataset path
- `dataset_path` parameter to `get_plantvillage_dataloaders()` for custom paths
- Auto-detection of class names from dataset folders
- Dynamic dataset path resolution

#### Modified:
- Dataset loading logic to use configurable paths
- Error messages to show dataset URL if available
- Class detection to update `config.PLANT_DISEASE_CLASSES` automatically

### 3. Main Script Updates (`main.py`)

#### Added:
- `select_dataset()` - Interactive dataset selection function
- `--dataset` command line argument for non-interactive selection
- Dataset selection logic in `main()` function
- Updated experiment header to show selected dataset name

#### Features:
- **Interactive Mode**: Prompts user to select from available datasets
- **Command Line Mode**: `--dataset 1-5` to specify dataset directly
- **Automatic Configuration**: Updates config variables based on selection

### 4. New Documentation Files

#### `DATASET_SELECTION.md`
Comprehensive guide covering:
- Available datasets with URLs and descriptions
- Dataset setup instructions
- How to select datasets (interactive and command line)
- Examples for different use cases
- Troubleshooting section

### 5. Test Script

#### `test_dataset_selection.py`
- Simple script to verify dataset configuration
- Lists all available datasets
- Useful for debugging

### 6. README Updates

Updated `README.md` to:
- Reflect multi-dataset support in project description
- Add dataset selection information in Quick Start
- Reference new `DATASET_SELECTION.md` guide
- Update project features to highlight dataset flexibility

## Usage Examples

### Interactive Selection
```powershell
python main.py --knn-zero-shot
# You'll be prompted to select a dataset
```

### Command Line Selection
```powershell
# Use PlantVillage dataset
python main.py --dataset 1 --all

# Use NEU Surface Defect dataset
python main.py --dataset 2 --linear-probe

# Use Semiconductor Wafer dataset
python main.py --dataset 4 --lora
```

### Test Configuration
```powershell
python test_dataset_selection.py
```

## How It Works

1. **Startup**: When `main.py` is run, it checks for `--dataset` argument
2. **Selection**: 
   - If `--dataset` is provided → Use that dataset
   - If not provided → Show interactive menu
3. **Configuration**: Selected dataset updates `config.CURRENT_DATASET`, `config.DATASET_NAME`, and `config.CURRENT_DATASET_PATH`
4. **Data Loading**: Data loader uses `get_current_dataset_path()` to load correct dataset
5. **Auto-Detection**: Class names are detected from folder structure
6. **Experiments**: All methods use the selected dataset

## Benefits

1. **Flexibility**: Easy to switch between datasets
2. **Scalability**: Easy to add new datasets to `config.AVAILABLE_DATASETS`
3. **User-Friendly**: Interactive mode with clear prompts
4. **Automation**: Classes are auto-detected
5. **Consistency**: All experiments use same dataset in single run
6. **Documentation**: Complete guides for setup and usage

## Future Enhancements

Possible additions:
- Dataset download automation
- Dataset validation checks
- Multi-dataset comparison mode
- Dataset statistics display
- Custom dataset addition wizard

## Files Modified

- `config.py` - Dataset configuration
- `data_loader.py` - Path handling and class detection
- `main.py` - Selection logic and argument parsing
- `README.md` - Documentation updates

## Files Created

- `DATASET_SELECTION.md` - Comprehensive dataset guide
- `test_dataset_selection.py` - Configuration test script

## Backward Compatibility

- Existing code continues to work
- Default behavior maintained (PlantVillage if no selection)
- All previous command line arguments still functional
- Data loader function signature extended (backward compatible)

## Testing Checklist

- [ ] Test interactive dataset selection
- [ ] Test command line dataset selection (--dataset 1-5)
- [ ] Test with actual datasets downloaded
- [ ] Verify class auto-detection
- [ ] Test all experiment types with different datasets
- [ ] Verify error messages when dataset not found
- [ ] Test backward compatibility

## Notes

- Datasets must be downloaded manually from Kaggle
- Each dataset must follow ImageFolder structure (class folders)
- Class names are taken from folder names
- All 5 datasets have been configured with Kaggle URLs
- System supports easy addition of new datasets
