# Quick Reference: Dataset Selection

## View Available Datasets
```powershell
python show_datasets.py
```

## Select Dataset When Running

### Interactive Mode (Easiest)
```powershell
python main.py --knn-zero-shot
```
You'll see a menu like this:
```
AVAILABLE DATASETS
======================================================================
1. PlantVillage Plant Disease
2. NEU Surface Defect Database
3. Goldenhar CFID
4. Multi-Class Semiconductor Wafer Image Dataset
5. PCB Defect (Modified)
======================================================================
Select a dataset (1-5):
```

### Command Line Mode (Faster)
```powershell
python main.py --dataset 1 --knn-zero-shot    # PlantVillage
python main.py --dataset 2 --linear-probe     # NEU Surface Defect
python main.py --dataset 3 --all              # Goldenhar CFID
python main.py --dataset 4 --lora             # Semiconductor Wafer
python main.py --dataset 5 --bitfit           # PCB Defect
```

## Dataset Setup

1. **Download** from Kaggle (see URLs in `show_datasets.py` output)
2. **Extract** to `./data/<dataset_name>/`
3. **Verify** structure: `data/dataset_name/class_name/*.jpg`

Example structure:
```
data/
├── plantvillage/
│   └── PlantVillage/
│       ├── Apple_scab/
│       ├── Apple_Black_rot/
│       └── ...
├── neu_surface_defect/
│   ├── crazing/
│   ├── patches/
│   └── ...
└── ...
```

## Experiment Examples

### Run all experiments on PlantVillage
```powershell
python main.py --dataset 1 --all
```

### Run KNN zero-shot on NEU Surface Defect
```powershell
python main.py --dataset 2 --knn-zero-shot
```

### Run adapter methods on Semiconductor Wafer
```powershell
python main.py --dataset 4 --adapters
```

### Run specific methods with interactive selection
```powershell
python main.py --linear-probe --lora --bitfit
# You'll be prompted to select dataset
```

## Tips

- Use `show_datasets.py` to see which datasets are installed
- Classes are auto-detected from folder names
- All results saved to `./results/` folder
- You can configure samples per class in `config.py` (default: 50)

## Common Issues

**Dataset not found?**
- Check path in `show_datasets.py` output
- Ensure folder structure matches (classes in subfolders)
- Download from Kaggle URL shown

**No classes detected?**
- Make sure images are in class subfolders, not at root
- Folder structure: `data/dataset/class_name/image.jpg`

## Dataset URLs

Run `python show_datasets.py` to see all Kaggle URLs and installation status.
