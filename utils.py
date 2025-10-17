"""
Utility functions for CLIP experiments
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import Dict, List
import torch

import config


def save_results(results: Dict, filename: str, results_dir: str = config.RESULTS_DIR):
    """
    Save results to a JSON file
    
    Args:
        results: Dictionary containing results
        filename: Name of the file to save
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    results_serializable[key][k] = v.tolist()
                elif isinstance(v, (list, dict)):
                    results_serializable[key][k] = v
                else:
                    results_serializable[key][k] = float(v) if isinstance(v, (int, float)) else v
        else:
            results_serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    
    print(f"Results saved to {filepath}")


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: str = None,
    top_n: int = 20
):
    """
    Plot confusion matrix
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
        top_n: Number of top classes to show (to avoid clutter)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # For large number of classes, show only top N most confused classes
    if len(class_names) > top_n:
        # Get top N classes based on total errors
        class_errors = cm.sum(axis=0) + cm.sum(axis=1) - 2 * np.diag(cm)
        top_indices = np.argsort(class_errors)[-top_n:]
        
        cm = cm[np.ix_(top_indices, top_indices)]
        class_names_subset = [class_names[i] for i in top_indices]
    else:
        class_names_subset = class_names
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names_subset,
        yticklabels=class_names_subset
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(
    history: Dict,
    save_path: str = None
):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def print_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: str = None
):
    """
    Print and optionally save classification report
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        class_names: List of class names
        save_path: Path to save the report
    """
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=3
    )
    
    print("\nClassification Report:")
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")


def compare_results(
    zero_shot_results: Dict,
    linear_probe_results: Dict,
    save_path: str = None
):
    """
    Create a comparison plot of zero-shot vs linear probing results
    
    Args:
        zero_shot_results: Results from zero-shot evaluation
        linear_probe_results: Results from linear probing
        save_path: Path to save the plot
    """
    splits = ['train', 'val', 'test']
    zero_shot_accs = [zero_shot_results[split]['accuracy'] for split in splits]
    linear_probe_accs = [linear_probe_results[split]['accuracy'] for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, zero_shot_accs, width, label='Zero-Shot', alpha=0.8)
    bars2 = ax.bar(x + width/2, linear_probe_accs, width, label='Linear Probe', alpha=0.8)
    
    ax.set_xlabel('Dataset Split')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Zero-Shot vs Linear Probing Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()


def get_device_info():
    """Print information about available compute devices"""
    print("="*50)
    print("DEVICE INFORMATION")
    print("="*50)
    
    if torch.cuda.is_available():
        print(f"CUDA is available")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total memory: {total_memory:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")
    
    print("="*50)


if __name__ == "__main__":
    # Test utility functions
    get_device_info()
