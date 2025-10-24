"""
Full fine-tuning of CLIP model on PlantVillage dataset
This method unfreezes all layers and trains the entire model end-to-end
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


class FullFineTuner:
    """
    Full fine-tuning of CLIP model
    """
    
    def __init__(
        self,
        model_name: str = config.CLIP_MODEL,
        pretrained: str = config.CLIP_PRETRAINED,
        num_classes: int = 102,
        device: torch.device = config.DEVICE,
        freeze_backbone: bool = False,
        freeze_text_encoder: bool = True
    ):
        """
        Initialize full fine-tuning model
        
        Args:
            model_name: Name of CLIP model architecture
            pretrained: Pretrained weights identifier
            num_classes: Number of classes in the dataset
            device: Device to run the model on
            freeze_backbone: Whether to freeze the backbone (partial fine-tuning)
            freeze_text_encoder: Whether to keep text encoder frozen
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
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            dummy_features = self.clip_model.encode_image(dummy_input)
            self.feature_dim = dummy_features.shape[-1]
        
        print(f"CLIP feature dimension: {self.feature_dim}")
        
        # Add classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes).to(device)
        
        # Configure which parts to train
        if freeze_backbone:
            print("Freezing backbone (image encoder)...")
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False
        else:
            print("Unfreezing all image encoder parameters...")
            for param in self.clip_model.visual.parameters():
                param.requires_grad = True
        
        if freeze_text_encoder:
            print("Freezing text encoder...")
            if hasattr(self.clip_model, 'transformer'):
                for param in self.clip_model.transformer.parameters():
                    param.requires_grad = False
            if hasattr(self.clip_model, 'token_embedding'):
                for param in self.clip_model.token_embedding.parameters():
                    param.requires_grad = False
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        print(f"Full fine-tuning initialized on {device}")
    
    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.clip_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        for _, param in self.classifier.named_parameters():
            all_params += param.numel()
            trainable_params += param.numel()
        
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable%: {100 * trainable_params / all_params:.2f}%")
    
    def forward(self, images):
        """Forward pass through model"""
        # Extract features
        features = self.clip_model.encode_image(images)
        features = F.normalize(features, dim=-1)
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def train_model(
        self,
        train_loader,
        val_loader,
        num_epochs: int = config.NUM_EPOCHS,
        learning_rate: float = 1e-5,  # Lower LR for fine-tuning
        weight_decay: float = 1e-4,
        patience: int = config.PATIENCE,
        warmup_epochs: int = 2
    ) -> dict:
        """
        Train the model with full fine-tuning
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            warmup_epochs: Number of warmup epochs
        
        Returns:
            Dictionary with training history
        """
        # Setup optimizer - separate LR for classifier and backbone
        backbone_params = []
        classifier_params = list(self.classifier.parameters())
        
        for name, param in self.clip_model.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
        
        if backbone_params:
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': learning_rate},
                {'params': classifier_params, 'lr': learning_rate * 10}  # Higher LR for classifier
            ], weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(classifier_params, lr=learning_rate * 10, weight_decay=weight_decay)
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * num_epochs
        warmup_steps = len(train_loader) * warmup_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
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
        
        print(f"\nTraining model for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            self.clip_model.train()
            self.classifier.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.clip_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * train_correct / train_total:.2f}%",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            self.clip_model.eval()
            self.classifier.eval()
            
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Val"):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.forward(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    predictions = outputs.argmax(dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
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
                best_state_dict = {
                    'clip_model': self.clip_model.state_dict(),
                    'classifier': self.classifier.state_dict()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        # Load best model
        if best_state_dict is not None:
            self.clip_model.load_state_dict(best_state_dict['clip_model'])
            self.classifier.load_state_dict(best_state_dict['classifier'])
            print(f"\nLoaded best model with validation accuracy: {best_val_acc:.2f}%")
        
        return history
    
    def evaluate(self, dataloader, split_name: str = "test") -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the trained model
        
        Args:
            dataloader: DataLoader for evaluation
            split_name: Name of the split being evaluated
        
        Returns:
            Tuple of (accuracy, predictions, ground_truth)
        """
        print(f"\nEvaluating on {split_name} set...")
        
        self.clip_model.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                images = images.to(self.device)
                
                outputs = self.forward(images)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                
                all_predictions.append(predictions)
                all_labels.append(labels.numpy())
        
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        accuracy = (all_predictions == all_labels).mean() * 100
        
        print(f"{split_name.capitalize()} Accuracy: {accuracy:.2f}%")
        
        return accuracy, all_predictions, all_labels
    
    def save_model(self, path: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'clip_model_state_dict': self.clip_model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.clip_model.load_state_dict(checkpoint['clip_model_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        print(f"Model loaded from {path}")


def run_full_finetuning(freeze_backbone: bool = False):
    """
    Main function to run full fine-tuning
    
    Args:
        freeze_backbone: If True, only train the classifier head
    """
    # Initialize fine-tuner
    finetuner = FullFineTuner(freeze_backbone=freeze_backbone)
    
    # Get data loaders with CLIP preprocessing
    train_loader, val_loader, test_loader = get_plantvillage_dataloaders(
        use_clip_transforms=True,
        clip_preprocess=finetuner.preprocess
    )
    
    # Train model
    history = finetuner.train_model(train_loader, val_loader)
    
    # Evaluate on all splits
    train_acc, train_preds, train_labels = finetuner.evaluate(train_loader, "train")
    val_acc, val_preds, val_labels = finetuner.evaluate(val_loader, "validation")
    test_acc, test_preds, test_labels = finetuner.evaluate(test_loader, "test")
    
    # Save model
    if config.SAVE_CHECKPOINT:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        model_name = "full_finetuning.pth" if not freeze_backbone else "head_only_finetuning.pth"
        finetuner.save_model(os.path.join(config.RESULTS_DIR, model_name))
    
    # Prepare results
    results = {
        'train': {'accuracy': train_acc, 'predictions': train_preds, 'labels': train_labels},
        'val': {'accuracy': val_acc, 'predictions': val_preds, 'labels': val_labels},
        'test': {'accuracy': test_acc, 'predictions': test_preds, 'labels': test_labels},
        'history': history
    }
    
    # Print summary
    print("\n" + "="*50)
    print("FULL FINE-TUNING RESULTS")
    print("="*50)
    print(f"Train Accuracy:      {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy:       {test_acc:.2f}%")
    print("="*50)
    
    return results


if __name__ == "__main__":
    results = run_full_finetuning()
