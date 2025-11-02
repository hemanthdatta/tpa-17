"""
Main script to run CLIP experiments on PlantVillage Crop Disease dataset
Implements:
1. Zero-shot evaluation (text-based and KNN-based)
2. Linear probing
3. Full fine-tuning
4. Adapter fine-tuning (LoRA, BitFit, Prefix-tuning)

Project Focus: Efficient fine-tuning of foundation models in low-data regimes
for domain-specific tasks (plant disease classification)
"""

import os
import argparse
import torch
from datetime import datetime

import config
from zero_shot import run_zero_shot_evaluation
from zero_shot_knn import run_knn_zero_shot_evaluation
from linear_probe import run_linear_probing
from full_finetuning import run_full_finetuning
from adapter_finetuning import run_adapter_finetuning
from utils import (
    save_results,
    plot_confusion_matrix,
    plot_training_history,
    print_classification_report,
    compare_results,
    get_device_info
)


def select_dataset():
    """
    Interactive dataset selection
    
    Returns:
        Selected dataset key
    """
    print("\n" + "="*70)
    print("AVAILABLE DATASETS")
    print("="*70)
    
    for key, dataset_info in config.AVAILABLE_DATASETS.items():
        print(f"{key}. {dataset_info['display_name']}")
        print(f"   URL: {dataset_info['url']}")
        print(f"   Path: {dataset_info['path']}")
        print()
    
    print("="*70)
    
    while True:
        try:
            choice = input("\nSelect a dataset (1-5): ").strip()
            if choice in config.AVAILABLE_DATASETS:
                selected = config.AVAILABLE_DATASETS[choice]
                print(f"\n✓ Selected: {selected['display_name']}")
                
                # Update config
                config.CURRENT_DATASET = choice
                config.DATASET_NAME = selected['name']
                config.CURRENT_DATASET_PATH = selected['path']
                
                return choice
            else:
                print("Invalid choice. Please select a number between 1 and 5.")
        except KeyboardInterrupt:
            print("\n\nDataset selection cancelled.")
            exit(0)
        except Exception as e:
            print(f"Error: {e}. Please try again.")


def run_experiments(args):
    """
    Run CLIP experiments on selected dataset
    
    Args:
        args: Command line arguments
    """
    # Get dataset info
    dataset_info = config.AVAILABLE_DATASETS[config.CURRENT_DATASET]
    
    print("\n" + "="*70)
    print(f"CLIP EXPERIMENTS ON {dataset_info['display_name'].upper()}")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_info['display_name']}")
    print(f"Dataset Path: {dataset_info['path']}")
    print(f"Model: {config.CLIP_MODEL}")
    print(f"Pretrained: {config.CLIP_PRETRAINED}")
    if config.USE_LIMITED_DATA:
        print(f"Low-Data Regime: {config.SAMPLES_PER_CLASS} samples/class")
    print("="*70 + "\n")
    
    # Show device info
    get_device_info()
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    zero_shot_results = None
    knn_zero_shot_results = None
    linear_probe_results = None
    full_finetune_results = None
    lora_results = None
    bitfit_results = None
    prefix_results = None
    
    # Run KNN zero-shot evaluation
    if args.knn_zero_shot:
        print("\n" + "="*70)
        print("RUNNING KNN ZERO-SHOT EVALUATION")
        print("="*70)
        
        knn_zero_shot_results = run_knn_zero_shot_evaluation(k_neighbors=args.k_neighbors)
        
        # Save results
        save_results(
            knn_zero_shot_results,
            "knn_zero_shot_results.json"
        )
        
        # Plot confusion matrix for test set
        plot_confusion_matrix(
            knn_zero_shot_results['test']['predictions'],
            knn_zero_shot_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            title=f"KNN Zero-Shot Classification (k={args.k_neighbors}) - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "knn_zero_shot_confusion_matrix.png")
        )
        
        # Print classification report
        print_classification_report(
            knn_zero_shot_results['test']['predictions'],
            knn_zero_shot_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            save_path=os.path.join(config.RESULTS_DIR, "knn_zero_shot_classification_report.txt")
        )
    
    # Run zero-shot evaluation (text-based)
    if args.zero_shot:
        print("\n" + "="*70)
        print("RUNNING TEXT-BASED ZERO-SHOT EVALUATION")
        print("="*70)
        
        zero_shot_results = run_zero_shot_evaluation()
        
        # Save results
        save_results(
            zero_shot_results,
            "zero_shot_results.json"
        )
        
        # Plot confusion matrix for test set
        plot_confusion_matrix(
            zero_shot_results['test']['predictions'],
            zero_shot_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            title="Text-Based Zero-Shot Classification - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "zero_shot_confusion_matrix.png")
        )
        
        # Print classification report
        print_classification_report(
            zero_shot_results['test']['predictions'],
            zero_shot_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            save_path=os.path.join(config.RESULTS_DIR, "zero_shot_classification_report.txt")
        )
    
    # Run linear probing
    if args.linear_probe:
        print("\n" + "="*70)
        print("RUNNING LINEAR PROBING")
        print("="*70)
        
        linear_probe_results = run_linear_probing()
        
        # Save results
        save_results(
            linear_probe_results,
            "linear_probe_results.json"
        )
        
        # Plot training history
        if 'history' in linear_probe_results:
            plot_training_history(
                linear_probe_results['history'],
                save_path=os.path.join(config.RESULTS_DIR, "linear_probe_training_history.png")
            )
        
        # Plot confusion matrix for test set
        plot_confusion_matrix(
            linear_probe_results['test']['predictions'],
            linear_probe_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            title="Linear Probing - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "linear_probe_confusion_matrix.png")
        )
        
        # Print classification report
        print_classification_report(
            linear_probe_results['test']['predictions'],
            linear_probe_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            save_path=os.path.join(config.RESULTS_DIR, "linear_probe_classification_report.txt")
        )
    
    # Run full fine-tuning
    if args.full_finetune:
        print("\n" + "="*70)
        print("RUNNING FULL FINE-TUNING")
        print("="*70)
        
        full_finetune_results = run_full_finetuning(freeze_backbone=False)
        
        # Save results
        save_results(
            full_finetune_results,
            "full_finetuning_results.json"
        )
        
        # Plot training history
        if 'history' in full_finetune_results:
            plot_training_history(
                full_finetune_results['history'],
                save_path=os.path.join(config.RESULTS_DIR, "full_finetuning_training_history.png")
            )
        
        # Plot confusion matrix for test set
        plot_confusion_matrix(
            full_finetune_results['test']['predictions'],
            full_finetune_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            title="Full Fine-tuning - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "full_finetuning_confusion_matrix.png")
        )
    
    # Run LoRA fine-tuning
    if args.lora:
        print("\n" + "="*70)
        print("RUNNING LORA FINE-TUNING")
        print("="*70)
        
        lora_results = run_adapter_finetuning(adapter_type="lora")
        
        # Save results
        save_results(lora_results, "lora_results.json")
        
        # Plot training history
        if 'history' in lora_results:
            plot_training_history(
                lora_results['history'],
                save_path=os.path.join(config.RESULTS_DIR, "lora_training_history.png")
            )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            lora_results['test']['predictions'],
            lora_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            title="LoRA Fine-tuning - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "lora_confusion_matrix.png")
        )
    
    # Run BitFit fine-tuning
    if args.bitfit:
        print("\n" + "="*70)
        print("RUNNING BITFIT FINE-TUNING")
        print("="*70)
        
        bitfit_results = run_adapter_finetuning(adapter_type="bitfit")
        
        # Save results
        save_results(bitfit_results, "bitfit_results.json")
        
        # Plot training history
        if 'history' in bitfit_results:
            plot_training_history(
                bitfit_results['history'],
                save_path=os.path.join(config.RESULTS_DIR, "bitfit_training_history.png")
            )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            bitfit_results['test']['predictions'],
            bitfit_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            title="BitFit Fine-tuning - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "bitfit_confusion_matrix.png")
        )
    
    # Run Prefix-tuning
    if args.prefix:
        print("\n" + "="*70)
        print("RUNNING PREFIX TUNING")
        print("="*70)
        
        prefix_results = run_adapter_finetuning(adapter_type="prefix")
        
        # Save results
        save_results(prefix_results, "prefix_results.json")
        
        # Plot training history
        if 'history' in prefix_results:
            plot_training_history(
                prefix_results['history'],
                save_path=os.path.join(config.RESULTS_DIR, "prefix_training_history.png")
            )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            prefix_results['test']['predictions'],
            prefix_results['test']['labels'],
            config.PLANT_DISEASE_CLASSES,
            title="Prefix-tuning - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "prefix_confusion_matrix.png")
        )
    
    # Print comprehensive comparison table
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*70)
    
    all_results = {}
    if knn_zero_shot_results is not None:
        all_results[f'KNN Zero-Shot (k={args.k_neighbors})'] = knn_zero_shot_results
    if zero_shot_results is not None:
        all_results['Text Zero-Shot'] = zero_shot_results
    if linear_probe_results is not None:
        all_results['Linear Probe'] = linear_probe_results
    if full_finetune_results is not None:
        all_results['Full Fine-tune'] = full_finetune_results
    if lora_results is not None:
        all_results['LoRA'] = lora_results
    if bitfit_results is not None:
        all_results['BitFit'] = bitfit_results
    if prefix_results is not None:
        all_results['Prefix-tuning'] = prefix_results
    
    if len(all_results) > 0:
        # Print table header
        print("\n{:<20} {:<12} {:<12} {:<12}".format(
            "Method", "Train", "Val", "Test"
        ))
        print("-" * 56)
        
        # Print results for each method
        for method_name, results in all_results.items():
            train_acc = results['train']['accuracy']
            val_acc = results['val']['accuracy']
            test_acc = results['test']['accuracy']
            
            print("{:<20} {:<12.2f}% {:<12.2f}% {:<12.2f}%".format(
                method_name, train_acc, val_acc, test_acc
            ))
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print("="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Foundation Model Fine-tuning on Multiple Datasets (Low-Data Regime)"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['1', '2', '3', '4', '5'],
        help='Dataset to use (1=PlantVillage, 2=NEU Surface Defect, 3=Goldenhar CFID, 4=Semiconductor Wafer, 5=PCB Defect)'
    )
    
    parser.add_argument(
        '--knn-zero-shot',
        action='store_true',
        help='Run KNN-based zero-shot evaluation'
    )
    
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=5,
        help='Number of neighbors for KNN zero-shot (default: 5)'
    )
    
    parser.add_argument(
        '--zero-shot',
        action='store_true',
        help='Run text-based zero-shot evaluation'
    )
    
    parser.add_argument(
        '--linear-probe',
        action='store_true',
        help='Run linear probing'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments (including both zero-shot methods)'
    )
    
    parser.add_argument(
        '--full-finetune',
        action='store_true',
        help='Run full fine-tuning'
    )
    
    parser.add_argument(
        '--lora',
        action='store_true',
        help='Run LoRA adapter fine-tuning'
    )
    
    parser.add_argument(
        '--bitfit',
        action='store_true',
        help='Run BitFit (bias-only) fine-tuning'
    )
    
    parser.add_argument(
        '--prefix',
        action='store_true',
        help='Run prefix-tuning'
    )
    
    parser.add_argument(
        '--adapters',
        action='store_true',
        help='Run all adapter methods (LoRA, BitFit, Prefix)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=config.CLIP_MODEL,
        help=f'CLIP model name (default: {config.CLIP_MODEL})'
    )
    
    parser.add_argument(
        '--pretrained',
        type=str,
        default=config.CLIP_PRETRAINED,
        help=f'Pretrained weights (default: {config.CLIP_PRETRAINED})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.BATCH_SIZE,
        help=f'Batch size (default: {config.BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.NUM_EPOCHS,
        help=f'Number of epochs for linear probing (default: {config.NUM_EPOCHS})'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=config.LEARNING_RATE,
        help=f'Learning rate for linear probing (default: {config.LEARNING_RATE})'
    )
    
    args = parser.parse_args()
    
    # Dataset selection
    if args.dataset:
        # Use dataset from command line argument
        if args.dataset in config.AVAILABLE_DATASETS:
            config.CURRENT_DATASET = args.dataset
            selected = config.AVAILABLE_DATASETS[args.dataset]
            config.DATASET_NAME = selected['name']
            config.CURRENT_DATASET_PATH = selected['path']
            print(f"\n✓ Using dataset: {selected['display_name']}")
        else:
            print(f"Error: Invalid dataset selection '{args.dataset}'")
            exit(1)
    else:
        # Interactive dataset selection
        select_dataset()
    
    # Update config with command line arguments
    config.CLIP_MODEL = args.model
    config.CLIP_PRETRAINED = args.pretrained
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    # If --all is specified, run all experiments
    if args.all:
        args.knn_zero_shot = True
        args.zero_shot = True
        args.linear_probe = True
        args.full_finetune = True
        args.lora = True
        args.bitfit = True
        args.prefix = True
    
    # If --adapters is specified, run all adapter methods
    if args.adapters:
        args.lora = True
        args.bitfit = True
        args.prefix = True
    
    # If no experiment is specified, run basic experiments by default
    if not any([args.knn_zero_shot, args.zero_shot, args.linear_probe, args.full_finetune, 
                args.lora, args.bitfit, args.prefix]):
        print("No experiment specified. Running KNN zero-shot and linear probing by default.")
        args.knn_zero_shot = True
        args.linear_probe = True
    
    # Run experiments
    run_experiments(args)


if __name__ == "__main__":
    main()
