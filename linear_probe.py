"""
Linear probing on CLIP features for PlantVillage dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from typing import Tuple, Optional
import numpy as np
import os

import config
from data_loader import get_plantvillage_dataloaders


class LinearClassifier(nn.Module):
    """
    Simple linear classifier for probing
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize linear classifier
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


class LinearProbe:
    """
    Linear probing on frozen CLIP features
    """
    
    def __init__(
        self,
        model_name: str = config.CLIP_MODEL,
        pretrained: str = config.CLIP_PRETRAINED,
        num_classes: int = 102,
        device: torch.device = config.DEVICE
    ):
        """
        Initialize linear probe
        
        Args:
            model_name: Name of CLIP model architecture
            pretrained: Pretrained weights identifier
            num_classes: Number of classes in the dataset
            device: Device to run the model on
        """
        self.device = device
        self.num_classes = num_classes
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name} with {pretrained} weights")
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()  # Freeze CLIP model
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_features = self.clip_model.encode_image(dummy_input)
            self.feature_dim = dummy_features.shape[-1]
        
        print(f"CLIP feature dimension: {self.feature_dim}")
        
        # Create linear classifier
        self.classifier = LinearClassifier(self.feature_dim, num_classes).to(device)
        
        print(f"Linear probe initialized on {device}")
    
    @torch.no_grad()
    def extract_features(self, dataloader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract CLIP features from a dataloader
        
        Args:
            dataloader: DataLoader to extract features from
        
        Returns:
            Tuple of (features, labels)
        """
        all_features = []
        all_labels = []
        
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(self.device)
            
            # Extract features
            features = self.clip_model.encode_image(images)
            features = F.normalize(features, dim=-1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
        
        # Concatenate all batches
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        
        return all_features, all_labels
    
    def train_classifier(
        self,
        train_loader,
        val_loader,
        num_epochs: int = config.NUM_EPOCHS,
        learning_rate: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        patience: int = config.PATIENCE
    ) -> dict:
        """
        Train the linear classifier on extracted features
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
        
        Returns:
            Dictionary with training history
        """
        print("\nExtracting features from training set...")
        train_features, train_labels = self.extract_features(train_loader)
        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)
        
        print("Extracting features from validation set...")
        val_features, val_labels = self.extract_features(val_loader)
        val_features = val_features.to(self.device)
        val_labels = val_labels.to(self.device)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        epochs_without_improvement = 0
        best_state_dict = None
        
        print(f"\nTraining linear classifier for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            self.classifier.train()
            
            # Create mini-batches for training
            batch_size = config.BATCH_SIZE
            num_samples = len(train_features)
            indices = torch.randperm(num_samples)
            
            train_loss = 0.0
            train_correct = 0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_features = train_features[batch_indices]
                batch_labels = train_labels[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                train_correct += (predictions == batch_labels).sum().item()
                num_batches += 1
            
            train_loss /= num_batches
            train_acc = train_correct / num_samples * 100
            
            # Validation
            self.classifier.eval()
            with torch.no_grad():
                val_outputs = self.classifier(val_features)
                val_loss = criterion(val_outputs, val_labels).item()
                val_predictions = val_outputs.argmax(dim=1)
                val_acc = (val_predictions == val_labels).float().mean().item() * 100
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = self.classifier.state_dict().copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        # Load best model
        if best_state_dict is not None:
            self.classifier.load_state_dict(best_state_dict)
            print(f"\nLoaded best model with validation accuracy: {best_val_acc:.2f}%")
        
        return history
    
    def evaluate(self, dataloader, split_name: str = "test") -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the trained classifier
        
        Args:
            dataloader: DataLoader for evaluation
            split_name: Name of the split being evaluated
        
        Returns:
            Tuple of (accuracy, predictions, ground_truth)
        """
        print(f"\nEvaluating on {split_name} set...")
        
        # Extract features
        features, labels = self.extract_features(dataloader)
        features = features.to(self.device)
        
        # Evaluate
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(features)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        
        labels = labels.numpy()
        accuracy = (predictions == labels).mean() * 100
        
        print(f"{split_name.capitalize()} Accuracy: {accuracy:.2f}%")
        
        return accuracy, predictions, labels
    
    def save_model(self, path: str):
        """Save the trained classifier"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained classifier"""
        checkpoint = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print(f"Model loaded from {path}")


def run_linear_probing():
    """
    Main function to run linear probing
    """
    # Initialize probe
    probe = LinearProbe()
    
    # Get data loaders with CLIP preprocessing
    train_loader, val_loader, test_loader = get_plantvillage_dataloaders(
        use_clip_transforms=True,
        clip_preprocess=probe.preprocess
    )
    
    # Train classifier
    history = probe.train_classifier(train_loader, val_loader)
    
    # Evaluate on all splits
    train_acc, train_preds, train_labels = probe.evaluate(train_loader, "train")
    val_acc, val_preds, val_labels = probe.evaluate(val_loader, "validation")
    test_acc, test_preds, test_labels = probe.evaluate(test_loader, "test")
    
    # Save model
    if config.SAVE_CHECKPOINT:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        probe.save_model(os.path.join(config.RESULTS_DIR, "linear_probe.pth"))
    
    # Prepare results
    results = {
        'train': {'accuracy': train_acc, 'predictions': train_preds, 'labels': train_labels},
        'val': {'accuracy': val_acc, 'predictions': val_preds, 'labels': val_labels},
        'test': {'accuracy': test_acc, 'predictions': test_preds, 'labels': test_labels},
        'history': history
    }
    
    # Print summary
    print("\n" + "="*50)
    print("LINEAR PROBING RESULTS")
    print("="*50)
    print(f"Train Accuracy:      {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy:       {test_acc:.2f}%")
    print("="*50)
    
    return results


if __name__ == "__main__":
    results = run_linear_probing()
