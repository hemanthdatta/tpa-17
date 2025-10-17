"""
Main script to run CLIP experiments on Flowers102 dataset
Implements:
1. Zero-shot evaluation
2. Linear probing
"""

import os
import argparse
import torch
from datetime import datetime

import config
from zero_shot import run_zero_shot_evaluation
from linear_probe import run_linear_probing
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
    
    # Compare results if both experiments were run
    if zero_shot_results is not None and linear_probe_results is not None:
        print("\n" + "="*70)
        print("COMPARISON: ZERO-SHOT VS LINEAR PROBING")
        print("="*70)
        
        compare_results(
            zero_shot_results,
            linear_probe_results,
            save_path=os.path.join(config.RESULTS_DIR, "comparison_plot.png")
        )
        
        # Print comparison table
        print("\n{:<15} {:<15} {:<15} {:<15}".format(
            "Split", "Zero-Shot", "Linear Probe", "Improvement"
        ))
        print("-" * 60)
        
        for split in ['train', 'val', 'test']:
            zs_acc = zero_shot_results[split]['accuracy']
            lp_acc = linear_probe_results[split]['accuracy']
            improvement = lp_acc - zs_acc
            
            print("{:<15} {:<15.2f}% {:<15.2f}% {:<15.2f}%".format(
                split.capitalize(), zs_acc, lp_acc, improvement
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
        help='Run all experiments (zero-shot and linear probing)'
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
    
    # If --all is specified, run both experiments
    if args.all:
        args.zero_shot = True
        args.linear_probe = True
    
    # If no experiment is specified, run all by default
    if not args.zero_shot and not args.linear_probe:
        print("No experiment specified. Running all experiments by default.")
        args.zero_shot = True
        args.linear_probe = True
    
    # Run experiments
    run_experiments(args)


if __name__ == "__main__":
    main()
