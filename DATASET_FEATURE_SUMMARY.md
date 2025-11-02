# 🎯 Dataset Selection Feature - Summary

## ✅ What's New

You can now **easily switch between different datasets** when running experiments!

### 🗂️ 5 Datasets Available

| # | Dataset | Domain | Use Case |
|---|---------|--------|----------|
| 1 | PlantVillage | Agriculture | Plant disease classification |
| 2 | NEU Surface Defect | Manufacturing | Steel defect detection |
| 3 | Goldenhar CFID | Medical | Medical imaging |
| 4 | Semiconductor Wafer | Electronics | Wafer defect detection |
| 5 | PCB Defect | Electronics | PCB quality control |

## 🚀 How to Use

### Method 1: Interactive (Recommended for First Time)
```powershell
python main.py --knn-zero-shot
```
▶️ You'll see a menu to select your dataset

### Method 2: Command Line (Faster)
```powershell
python main.py --dataset 1 --all
```
▶️ Directly uses dataset #1 (PlantVillage)

### Method 3: Check What's Installed
```powershell
python show_datasets.py
```
▶️ Shows which datasets you have downloaded

## 📋 Complete Workflow

```
1. Check available datasets
   └─ python show_datasets.py

2. Download dataset from Kaggle
   └─ Visit URL shown in output
   └─ Download and extract

3. Run experiments
   └─ python main.py --dataset <number> --<experiment>
```

## 📂 Dataset Structure Required

```
data/
└── <dataset_name>/
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/
    │   └── ...
    └── ...
```

## 💡 Examples

```powershell
# View datasets
python show_datasets.py

# Run KNN zero-shot on PlantVillage
python main.py --dataset 1 --knn-zero-shot

# Run linear probing on NEU Surface Defect
python main.py --dataset 2 --linear-probe

# Run all adapter methods on Semiconductor Wafer
python main.py --dataset 4 --adapters

# Interactive selection with multiple experiments
python main.py --linear-probe --lora --bitfit
```

## 📚 Documentation Files

| File | Description |
|------|-------------|
| `DATASET_SELECTION.md` | Complete guide with setup instructions |
| `QUICK_DATASET_REFERENCE.md` | Quick commands and tips |
| `show_datasets.py` | Script to view available datasets |
| `test_dataset_selection.py` | Test configuration |

## 🎨 Features

✨ **Auto-Detection**: Classes detected from folder names  
✨ **Flexible**: Easy to add new datasets  
✨ **User-Friendly**: Interactive prompts  
✨ **Documented**: Complete setup guides  
✨ **Backward Compatible**: Old code still works  

## 🔧 Configuration

Edit `config.py` to:
- Add new datasets to `AVAILABLE_DATASETS`
- Change samples per class (default: 50)
- Modify other hyperparameters

## ⚡ Quick Commands Cheat Sheet

```powershell
# Show datasets with installation status
python show_datasets.py

# Interactive dataset selection
python main.py --knn-zero-shot

# Use specific dataset
python main.py --dataset <1-5> --<experiment>

# Test configuration
python test_dataset_selection.py
```

## 🎓 All Experiment Options

```powershell
--knn-zero-shot     # KNN-based zero-shot
--zero-shot         # Text-based zero-shot
--linear-probe      # Linear probing
--full-finetune     # Full fine-tuning
--lora              # LoRA adapter
--bitfit            # BitFit adapter
--prefix            # Prefix tuning
--adapters          # All adapters
--all               # All experiments
```

## 📞 Need Help?

- See `DATASET_SELECTION.md` for detailed setup
- Run `python show_datasets.py` to check installation
- Check `QUICK_DATASET_REFERENCE.md` for commands
