"""
Main script to run CLIP experiments on Flowers102 dataset
Implements:
1. Zero-shot evaluation
2. Linear probing
3. Full fine-tuning
4. Adapter fine-tuning (LoRA, BitFit, Prefix-tuning)
"""

import os
import argparse
import torch
from datetime import datetime

import config
from zero_shot import run_zero_shot_evaluation
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


def run_experiments(args):
    """
    Run CLIP experiments on Flowers102 dataset
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*70)
    print("CLIP EXPERIMENTS ON FLOWERS102 DATASET")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {config.CLIP_MODEL}")
    print(f"Pretrained: {config.CLIP_PRETRAINED}")
    print("="*70 + "\n")
    
    # Show device info
    get_device_info()
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    zero_shot_results = None
    linear_probe_results = None
    full_finetune_results = None
    lora_results = None
    bitfit_results = None
    prefix_results = None
    
    # Run zero-shot evaluation
    if args.zero_shot:
        print("\n" + "="*70)
        print("RUNNING ZERO-SHOT EVALUATION")
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
            config.FLOWER_CLASSES,
            title="Zero-Shot Classification - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "zero_shot_confusion_matrix.png")
        )
        
        # Print classification report
        print_classification_report(
            zero_shot_results['test']['predictions'],
            zero_shot_results['test']['labels'],
            config.FLOWER_CLASSES,
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
            config.FLOWER_CLASSES,
            title="Linear Probing - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "linear_probe_confusion_matrix.png")
        )
        
        # Print classification report
        print_classification_report(
            linear_probe_results['test']['predictions'],
            linear_probe_results['test']['labels'],
            config.FLOWER_CLASSES,
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
            config.FLOWER_CLASSES,
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
            config.FLOWER_CLASSES,
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
            config.FLOWER_CLASSES,
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
            config.FLOWER_CLASSES,
            title="Prefix-tuning - Test Set",
            save_path=os.path.join(config.RESULTS_DIR, "prefix_confusion_matrix.png")
        )
    
    # Print comprehensive comparison table
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*70)
    
    all_results = {}
    if zero_shot_results is not None:
        all_results['Zero-Shot'] = zero_shot_results
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
        description="CLIP experiments on Flowers102 dataset"
    )
    
    parser.add_argument(
        '--zero-shot',
        action='store_true',
        help='Run zero-shot evaluation'
    )
    
    parser.add_argument(
        '--linear-probe',
        action='store_true',
        help='Run linear probing'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
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
    
    # Update config with command line arguments
    config.CLIP_MODEL = args.model
    config.CLIP_PRETRAINED = args.pretrained
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    
    # If --all is specified, run all experiments
    if args.all:
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
    if not any([args.zero_shot, args.linear_probe, args.full_finetune, 
                args.lora, args.bitfit, args.prefix]):
        print("No experiment specified. Running zero-shot and linear probing by default.")
        args.zero_shot = True
        args.linear_probe = True
    
    # Run experiments
    run_experiments(args)


if __name__ == "__main__":
    main()
