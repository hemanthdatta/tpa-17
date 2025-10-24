"""
KNN-based zero-shot classification using CLIP on PlantVillage dataset
This approach uses K-Nearest Neighbors in the feature space instead of text templates
"""

import torch
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from typing import Tuple, Optional
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

import config
from data_loader import get_plantvillage_dataloaders


class KNNZeroShotClassifier:
    """
    KNN-based zero-shot classifier using CLIP features
    Instead of using text templates, this approach:
    1. Extracts visual features from a small support set
    2. Uses KNN to classify test images based on visual similarity
    """
    
    def __init__(
        self,
        model_name: str = config.CLIP_MODEL,
        pretrained: str = config.CLIP_PRETRAINED,
        device: torch.device = config.DEVICE,
        k_neighbors: int = 5
    ):
        """
        Initialize the KNN zero-shot classifier
        
        Args:
            model_name: Name of CLIP model architecture
            pretrained: Pretrained weights identifier
            device: Device to run the model on
            k_neighbors: Number of neighbors for KNN
        """
        self.device = device
        self.k_neighbors = k_neighbors
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name} with {pretrained} weights")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # KNN classifier will be initialized after extracting support features
        self.knn_classifier = None
        
        print(f"Model loaded successfully on {device}")
        print(f"KNN configuration: k={k_neighbors}")
    
    def extract_features(
        self,
        dataloader,
        split_name: str = "support"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from images
        
        Args:
            dataloader: DataLoader for the dataset
            split_name: Name of the split (for logging)
        
        Returns:
            Tuple of (features, labels)
        """
        print(f"Extracting features from {split_name} set...")
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Extracting {split_name} features"):
                images = images.to(self.device)
                
                # Extract image features
                image_features = self.model.encode_image(images)
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                
                all_features.append(image_features.cpu())
                all_labels.append(labels)
        
        # Concatenate all features and labels
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        print(f"Extracted {len(features)} features from {split_name} set")
        
        return features, labels
    
    def fit_knn(
        self,
        support_loader,
        metric: str = 'cosine'
    ):
        """
        Fit KNN classifier on support set features
        
        Args:
            support_loader: DataLoader for support set (typically train set with limited samples)
            metric: Distance metric for KNN ('cosine', 'euclidean', 'manhattan')
        """
        print("\n" + "="*70)
        print("FITTING KNN CLASSIFIER ON SUPPORT SET")
        print("="*70)
        
        # Extract features from support set
        support_features, support_labels = self.extract_features(
            support_loader, 
            split_name="support"
        )
        
        # Convert to numpy for sklearn
        support_features_np = support_features.numpy()
        support_labels_np = support_labels.numpy()
        
        # Initialize and fit KNN classifier
        print(f"\nTraining KNN classifier (k={self.k_neighbors}, metric={metric})...")
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=self.k_neighbors,
            metric=metric,
            weights='distance',  # Weight by inverse distance
            algorithm='brute',   # Use brute force for cosine similarity
            n_jobs=-1            # Use all CPU cores
        )
        
        self.knn_classifier.fit(support_features_np, support_labels_np)
        
        # Calculate support set accuracy (sanity check)
        support_acc = self.knn_classifier.score(support_features_np, support_labels_np)
        
        print(f"KNN classifier fitted successfully!")
        print(f"Support set accuracy: {support_acc * 100:.2f}%")
        print("="*70 + "\n")
    
    def evaluate_knn(
        self,
        dataloader,
        split_name: str = "test"
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate KNN classification on a dataset
        
        Args:
            dataloader: DataLoader for the dataset
            split_name: Name of the split (for logging)
        
        Returns:
            Tuple of (accuracy, predictions, true_labels)
        """
        if self.knn_classifier is None:
            raise ValueError("KNN classifier not fitted. Call fit_knn() first.")
        
        print(f"Evaluating on {split_name} set using KNN...")
        
        # Extract features
        features, labels = self.extract_features(dataloader, split_name=split_name)
        
        # Convert to numpy
        features_np = features.numpy()
        labels_np = labels.numpy()
        
        # Predict using KNN
        predictions = self.knn_classifier.predict(features_np)
        
        # Calculate accuracy
        accuracy = (predictions == labels_np).mean() * 100
        
        print(f"{split_name.capitalize()} Accuracy: {accuracy:.2f}%")
        
        return accuracy, predictions, labels_np
    
    def evaluate_with_confidence(
        self,
        dataloader,
        split_name: str = "test"
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate KNN classification with confidence scores
        
        Args:
            dataloader: DataLoader for the dataset
            split_name: Name of the split (for logging)
        
        Returns:
            Tuple of (accuracy, predictions, true_labels, confidence_scores)
        """
        if self.knn_classifier is None:
            raise ValueError("KNN classifier not fitted. Call fit_knn() first.")
        
        print(f"Evaluating on {split_name} set using KNN (with confidence)...")
        
        # Extract features
        features, labels = self.extract_features(dataloader, split_name=split_name)
        
        # Convert to numpy
        features_np = features.numpy()
        labels_np = labels.numpy()
        
        # Predict with probability
        predictions = self.knn_classifier.predict(features_np)
        probabilities = self.knn_classifier.predict_proba(features_np)
        
        # Confidence is the max probability
        confidence_scores = probabilities.max(axis=1)
        
        # Calculate accuracy
        accuracy = (predictions == labels_np).mean() * 100
        
        print(f"{split_name.capitalize()} Accuracy: {accuracy:.2f}%")
        print(f"Average confidence: {confidence_scores.mean():.4f}")
        
        return accuracy, predictions, labels_np, confidence_scores
    
    def evaluate_all_splits(
        self,
        train_loader,
        val_loader,
        test_loader
    ) -> dict:
        """
        Evaluate KNN classification on all splits
        
        Args:
            train_loader: Training data loader (used as support set)
            val_loader: Validation data loader
            test_loader: Test data loader
        
        Returns:
            Dictionary with results for all splits
        """
        # First, fit KNN on training set
        self.fit_knn(train_loader)
        
        # Evaluate on all splits
        train_acc, train_preds, train_labels = self.evaluate_knn(
            train_loader, "train"
        )
        val_acc, val_preds, val_labels = self.evaluate_knn(
            val_loader, "validation"
        )
        test_acc, test_preds, test_labels = self.evaluate_knn(
            test_loader, "test"
        )
        
        results = {
            'train': {
                'accuracy': train_acc,
                'predictions': train_preds,
                'labels': train_labels
            },
            'val': {
                'accuracy': val_acc,
                'predictions': val_preds,
                'labels': val_labels
            },
            'test': {
                'accuracy': test_acc,
                'predictions': test_preds,
                'labels': test_labels
            }
        }
        
        return results


def run_knn_zero_shot_evaluation(k_neighbors: int = 5):
    """
    Main function to run KNN-based zero-shot evaluation on PlantVillage dataset
    
    Args:
        k_neighbors: Number of neighbors for KNN
    """
    # Initialize classifier
    classifier = KNNZeroShotClassifier(k_neighbors=k_neighbors)
    
    # Get data loaders with CLIP preprocessing
    train_loader, val_loader, test_loader = get_plantvillage_dataloaders(
        use_clip_transforms=True,
        clip_preprocess=classifier.preprocess
    )
    
    # Run evaluation
    results = classifier.evaluate_all_splits(
        train_loader, val_loader, test_loader
    )
    
    # Print summary
    print("\n" + "="*70)
    print("KNN ZERO-SHOT CLASSIFICATION RESULTS")
    print("="*70)
    print(f"K-Neighbors: {k_neighbors}")
    print(f"Train Accuracy:      {results['train']['accuracy']:.2f}%")
    print(f"Validation Accuracy: {results['val']['accuracy']:.2f}%")
    print(f"Test Accuracy:       {results['test']['accuracy']:.2f}%")
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Test the KNN zero-shot classifier
    import argparse
    
    parser = argparse.ArgumentParser(
        description="KNN-based zero-shot classification on PlantVillage"
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of neighbors for KNN (default: 5)'
    )
    
    args = parser.parse_args()
    
    results = run_knn_zero_shot_evaluation(k_neighbors=args.k)
