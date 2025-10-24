"""
Zero-shot classification using CLIP on PlantVillage dataset
"""

import torch
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from typing import List, Tuple
import numpy as np

import config
from data_loader import get_plantvillage_dataloaders


class ZeroShotClassifier:
    """
    Zero-shot classifier using CLIP model
    """
    
    def __init__(
        self,
        model_name: str = config.CLIP_MODEL,
        pretrained: str = config.CLIP_PRETRAINED,
        device: torch.device = config.DEVICE
    ):
        """
        Initialize the zero-shot classifier
        
        Args:
            model_name: Name of CLIP model architecture
            pretrained: Pretrained weights identifier
            device: Device to run the model on
        """
        self.device = device
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name} with {pretrained} weights")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully on {device}")
    
    def create_text_features(
        self,
        class_names: List[str],
        templates: List[str] = config.TEXT_TEMPLATES
    ) -> torch.Tensor:
        """
        Create text embeddings for class names using multiple templates
        
        Args:
            class_names: List of class names
            templates: List of text templates
        
        Returns:
            Text features tensor of shape (num_classes, feature_dim)
        """
        print("Creating text embeddings for classes...")
        
        all_text_features = []
        
        with torch.no_grad():
            for class_name in tqdm(class_names, desc="Processing classes"):
                # Create texts from templates
                texts = [template.format(class_name) for template in templates]
                
                # Tokenize
                text_tokens = self.tokenizer(texts).to(self.device)
                
                # Get text features
                text_features = self.model.encode_text(text_tokens)
                
                # Average across templates and normalize
                text_features = text_features.mean(dim=0)
                text_features = F.normalize(text_features, dim=-1)
                
                all_text_features.append(text_features)
        
        # Stack all text features
        text_features = torch.stack(all_text_features)
        
        return text_features
    
    def evaluate(
        self,
        dataloader,
        text_features: torch.Tensor,
        split_name: str = "test"
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate zero-shot classification on a dataset
        
        Args:
            dataloader: DataLoader for the dataset
            text_features: Pre-computed text features for all classes
            split_name: Name of the split being evaluated
        
        Returns:
            Tuple of (accuracy, predictions, ground_truth)
        """
        print(f"\nEvaluating zero-shot classification on {split_name} set...")
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get image features
                image_features = self.model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                
                # Compute similarity with text features
                # Shape: (batch_size, num_classes)
                logits = image_features @ text_features.T
                
                # Get predictions
                predictions = logits.argmax(dim=-1)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # Calculate accuracy
        accuracy = (all_predictions == all_labels).mean() * 100
        
        print(f"{split_name.capitalize()} Accuracy: {accuracy:.2f}%")
        
        return accuracy, all_predictions, all_labels
    
    def evaluate_all_splits(
        self,
        train_loader,
        val_loader,
        test_loader,
        class_names: List[str] = config.PLANT_DISEASE_CLASSES
    ) -> dict:
        """
        Evaluate zero-shot classification on all splits
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            class_names: List of class names (plant disease classes)
        
        Returns:
            Dictionary with results for all splits
        """
        # Create text features once
        text_features = self.create_text_features(class_names)
        
        results = {}
        
        # Evaluate on train set
        train_acc, train_preds, train_labels = self.evaluate(
            train_loader, text_features, "train"
        )
        results['train'] = {
            'accuracy': train_acc,
            'predictions': train_preds,
            'labels': train_labels
        }
        
        # Evaluate on validation set
        val_acc, val_preds, val_labels = self.evaluate(
            val_loader, text_features, "validation"
        )
        results['val'] = {
            'accuracy': val_acc,
            'predictions': val_preds,
            'labels': val_labels
        }
        
        # Evaluate on test set
        test_acc, test_preds, test_labels = self.evaluate(
            test_loader, text_features, "test"
        )
        results['test'] = {
            'accuracy': test_acc,
            'predictions': test_preds,
            'labels': test_labels
        }
        
        return results


def run_zero_shot_evaluation():
    """
    Main function to run zero-shot evaluation on PlantVillage dataset
    """
    # Initialize classifier
    classifier = ZeroShotClassifier()
    
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
    print("\n" + "="*50)
    print("ZERO-SHOT CLASSIFICATION RESULTS")
    print("="*50)
    print(f"Train Accuracy:      {results['train']['accuracy']:.2f}%")
    print(f"Validation Accuracy: {results['val']['accuracy']:.2f}%")
    print(f"Test Accuracy:       {results['test']['accuracy']:.2f}%")
    print("="*50)
    
    return results


if __name__ == "__main__":
    results = run_zero_shot_evaluation()
