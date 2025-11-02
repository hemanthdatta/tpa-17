# Dataset Selection Guide

This project supports multiple datasets for CLIP-based experiments. You can select which dataset to use when running `main.py`.

## Available Datasets

### 1. PlantVillage Plant Disease
- **URL**: https://www.kaggle.com/datasets/emmarex/plantdisease
- **Path**: `data/plantvillage/PlantVillage`
- **Description**: Plant disease classification dataset with 38 classes

### 2. NEU Surface Defect Database
- **URL**: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
- **Path**: `data/neu_surface_defect`
- **Description**: Steel surface defect detection dataset

### 3. Goldenhar CFID
- **URL**: https://www.kaggle.com/datasets/isratjahan123/goldenhar-cfid
- **Path**: `data/goldenhar_cfid`
- **Description**: Medical imaging dataset for Goldenhar syndrome

### 4. Multi-Class Semiconductor Wafer Image Dataset
- **URL**: https://www.kaggle.com/datasets/drtawfikrahman/multi-class-semiconductor-wafer-image-dataset
- **Path**: `data/semiconductor_wafer`
- **Description**: Semiconductor wafer defect classification

### 5. PCB Defect (Modified)
- **URL**: https://www.kaggle.com/datasets/breaddddd/pcb-defect-modified
- **Path**: `data/pcb_defect`
- **Description**: PCB defect detection dataset

## Dataset Setup Instructions

### Step 1: Download Dataset from Kaggle

1. Go to the dataset URL
2. Click "Download" button
3. Extract the downloaded ZIP file

### Step 2: Organize Dataset Structure

The dataset should be organized in the following structure:
```
data/
├── plantvillage/
│   └── PlantVillage/
│       ├── class_1/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       ├── class_2/
│       └── ...
├── neu_surface_defect/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── goldenhar_cfid/
│   ├── class_1/
│   └── ...
├── semiconductor_wafer/
│   ├── class_1/
│   └── ...
└── pcb_defect/
    ├── class_1/
    └── ...
```

**Important**: Each dataset folder should contain subfolders for each class, and each class folder should contain the images for that class.

### Step 3: Verify Dataset Structure

You can test if the dataset is correctly set up by running:
```powershell
python data_loader.py
```

This will attempt to load the default dataset and show information about the classes and samples.

## How to Select a Dataset

### Method 1: Interactive Selection (Recommended)

Simply run `main.py` without the `--dataset` argument:

```powershell
python main.py --knn-zero-shot
```

You'll be prompted to select a dataset:
```
AVAILABLE DATASETS
======================================================================
1. PlantVillage Plant Disease
   URL: https://www.kaggle.com/datasets/emmarex/plantdisease
   Path: plantvillage/PlantVillage

2. NEU Surface Defect Database
   URL: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
   Path: neu_surface_defect

...

Select a dataset (1-5): 
```

### Method 2: Command Line Argument

Specify the dataset using the `--dataset` argument:

```powershell
# Use PlantVillage dataset
python main.py --dataset 1 --knn-zero-shot

# Use NEU Surface Defect dataset
python main.py --dataset 2 --linear-probe

# Use Semiconductor Wafer dataset
python main.py --dataset 4 --all
```

## Examples

### Run KNN Zero-Shot with PlantVillage Dataset
```powershell
python main.py --dataset 1 --knn-zero-shot
```

### Run All Experiments with NEU Surface Defect Dataset
```powershell
python main.py --dataset 2 --all
```

### Run LoRA Fine-tuning with PCB Defect Dataset
```powershell
python main.py --dataset 5 --lora
```

### Interactive Selection with Multiple Experiments
```powershell
python main.py --linear-probe --lora --bitfit
# You'll be prompted to select a dataset
```

## Troubleshooting

### Dataset Not Found Error

If you see an error like:
```
DATASET NOT FOUND
Expected location: c:\Users\heman\Desktop\tpa-17\data\neu_surface_defect
```

**Solution**:
1. Download the dataset from the provided URL
2. Extract it to the correct location
3. Ensure the folder structure matches (each class in its own subfolder)

### Class Detection Issues

The system automatically detects classes from folder names. Make sure:
- Each class has its own folder
- Folder names are meaningful (they will be used as class labels)
- There are no empty folders

### Path Issues on Windows

If you have path-related issues:
- Use forward slashes `/` or double backslashes `\\` in paths
- Avoid spaces in folder names
- Use absolute paths if relative paths don't work

## Notes

- The system automatically detects the number of classes in your dataset
- Class names are taken from the folder names
- All experiments use the same dataset for a single run
- Results are saved in the `results/` folder with dataset-specific names
