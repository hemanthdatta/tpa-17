"""
Adapter-based fine-tuning methods for CLIP on Flowers102 dataset
Includes: LoRA, BitFit, and Prefix-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
from typing import Tuple, Optional, List
import numpy as np
import os
import math

import config
from data_loader import get_flowers_dataloaders


# ============================================================================
# LoRA Implementation
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer
    Adds trainable low-rank matrices to frozen weights
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize A with kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """Forward pass through LoRA layer"""
        # x @ A @ B with scaling
        lora_out = (x @ self.lora_A) @ self.lora_B
        lora_out = self.lora_dropout(lora_out)
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation
    """
    
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank,
            alpha,
            dropout
        )
        
        # Store dimensions for compatibility
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    @property
    def weight(self):
        """Expose weight attribute for compatibility with MultiheadAttention"""
        return self.linear.weight
    
    @property
    def bias(self):
        """Expose bias attribute for compatibility with MultiheadAttention"""
        return self.linear.bias
    
    def forward(self, x):
        """Forward pass: frozen linear + LoRA"""
        return self.linear(x) + self.lora(x)


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, target_modules: List[str] = None):
    """
    Inject LoRA layers into a model
    
    Args:
        model: Model to inject LoRA into
        rank: Rank of LoRA matrices
        alpha: LoRA alpha parameter
        target_modules: List of module names to target
    
    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        # Target MLP layers in OpenCLIP Vision Transformer
        # These are safer to replace than attention projections
        target_modules = ['mlp.c_fc', 'mlp.c_proj']
    
    lora_modules_added = 0
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Check if this module should be adapted
            # Use exact matching for better control
            module_type = name.split('.')[-1] if '.' in name else name
            parent_context = '.'.join(name.split('.')[-2:]) if len(name.split('.')) >= 2 else name
            
            should_adapt = any(target in parent_context for target in target_modules)
            
            if should_adapt:
                try:
                    # Get parent module and attribute name
                    *parent_path, attr_name = name.split('.')
                    parent = model
                    for p in parent_path:
                        parent = getattr(parent, p)
                    
                    # Replace with LoRA linear
                    lora_linear = LoRALinear(module, rank, alpha)
                    setattr(parent, attr_name, lora_linear)
                    lora_modules_added += 1
                except Exception as e:
                    print(f"Warning: Could not inject LoRA into {name}: {e}")
                    continue
    
    print(f"✓ Injected LoRA into {lora_modules_added} modules (rank={rank}, alpha={alpha})")
    
    if lora_modules_added == 0:
        print("⚠ Warning: No LoRA modules were added! Check target_modules pattern.")
    
    return model


# ============================================================================
# BitFit Implementation
# ============================================================================

def apply_bitfit(model: nn.Module):
    """
    Apply BitFit: Only train bias terms
    
    Args:
        model: Model to apply BitFit to
    
    Returns:
        Modified model with only bias terms trainable
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only bias terms
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
    
    return model


# ============================================================================
# Prefix Tuning Implementation
# ============================================================================

class PrefixEncoder(nn.Module):
    """
    Prefix encoder for prefix-tuning
    Learns continuous prompts prepended to the input
    """
    
    def __init__(
        self,
        num_virtual_tokens: int = 10,
        hidden_size: int = 768,
        num_layers: int = 12
    ):
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Prefix parameters for each layer
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_layers, num_virtual_tokens, hidden_size)
        )
        
        # MLP to project prefix tokens
        self.prefix_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, batch_size: int):
        """
        Generate prefix for the batch
        
        Args:
            batch_size: Size of the batch
        
        Returns:
            Prefix tensor of shape (num_layers, batch_size, num_virtual_tokens, hidden_size)
        """
        # Expand prefix tokens for batch
        prefix = self.prefix_tokens.unsqueeze(1).expand(-1, batch_size, -1, -1)
        
        # Apply MLP transformation
        prefix = self.prefix_mlp(prefix)
        
        return prefix


def inject_prefix_tuning(model: nn.Module, num_virtual_tokens: int = 10, hidden_size: int = 768):
    """
    Inject prefix tuning into a vision transformer
    
    Args:
        model: Model to inject prefix tuning into
        num_virtual_tokens: Number of virtual tokens to prepend
        hidden_size: Hidden size of the model
    
    Returns:
        Model with prefix tuning and the prefix encoder
    """
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Count number of transformer layers
    num_layers = 0
    for name, module in model.named_modules():
        if 'blocks' in name or 'layers' in name:
            if isinstance(module, nn.ModuleList):
                num_layers = len(module)
                break
    
    if num_layers == 0:
        num_layers = 12  # Default
    
    # Create prefix encoder
    prefix_encoder = PrefixEncoder(num_virtual_tokens, hidden_size, num_layers)
    
    return model, prefix_encoder


# ============================================================================
# Unified Adapter Fine-tuner
# ============================================================================

class AdapterFineTuner:
    """
    Fine-tuning with adapter methods (LoRA, BitFit, Prefix-tuning)
    """
    
    def __init__(
        self,
        model_name: str = config.CLIP_MODEL,
        pretrained: str = config.CLIP_PRETRAINED,
        num_classes: int = 102,
        device: torch.device = config.DEVICE,
        adapter_type: str = "lora",  # "lora", "bitfit", "prefix"
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        num_prefix_tokens: int = 10
    ):
        """
        Initialize adapter fine-tuner
        
        Args:
            model_name: Name of CLIP model architecture
            pretrained: Pretrained weights identifier
            num_classes: Number of classes in the dataset
            device: Device to run the model on
            adapter_type: Type of adapter ("lora", "bitfit", "prefix")
            lora_rank: Rank for LoRA adaptation
            lora_alpha: Alpha for LoRA adaptation
            num_prefix_tokens: Number of prefix tokens for prefix-tuning
        """
        self.device = device
        self.num_classes = num_classes
        self.adapter_type = adapter_type
        
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
        
        # Apply adapter method
        self.prefix_encoder = None
        if adapter_type == "lora":
            print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
            # CRITICAL: Freeze entire backbone first
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # Then inject LoRA (which will have requires_grad=True)
            self.clip_model.visual = inject_lora(
                self.clip_model.visual,
                rank=lora_rank,
                alpha=lora_alpha
            )
            # Move model to device after LoRA injection
            self.clip_model = self.clip_model.to(device)
        elif adapter_type == "bitfit":
            print("Applying BitFit (bias-only training)")
            self.clip_model.visual = apply_bitfit(self.clip_model.visual)
        elif adapter_type == "prefix":
            print(f"Applying Prefix-tuning with {num_prefix_tokens} tokens")
            # CRITICAL: Freeze entire backbone first
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # Then inject prefix-tuning
            self.clip_model.visual, self.prefix_encoder = inject_prefix_tuning(
                self.clip_model.visual,
                num_prefix_tokens,
                self.feature_dim
            )
            if self.prefix_encoder is not None:
                self.prefix_encoder = self.prefix_encoder.to(device)
            # Move model to device after prefix injection
            self.clip_model = self.clip_model.to(device)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        # Add classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes).to(device)
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
        print(f"Adapter fine-tuning ({adapter_type}) initialized on {device}")
    
    def _print_trainable_parameters(self):
        """Print number of trainable parameters"""
        trainable_params = 0
        all_params = 0
        lora_params = 0
        
        for name, param in self.clip_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'lora' in name.lower():
                    lora_params += param.numel()
        
        for _, param in self.classifier.named_parameters():
            all_params += param.numel()
            trainable_params += param.numel()
        
        if self.prefix_encoder is not None:
            for _, param in self.prefix_encoder.named_parameters():
                all_params += param.numel()
                trainable_params += param.numel()
        
        print(f"\nTrainable Parameters:")
        print(f"  Total: {all_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"  Frozen: {all_params - trainable_params:,} ({100 * (all_params - trainable_params) / all_params:.2f}%)")
        if lora_params > 0:
            print(f"  LoRA params: {lora_params:,}")
        print()
    
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
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = config.PATIENCE
    ) -> dict:
        """
        Train the model with adapter
        
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
        # Setup optimizer
        trainable_params = []
        for param in self.clip_model.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        trainable_params.extend(self.classifier.parameters())
        
        if self.prefix_encoder is not None:
            trainable_params.extend(self.prefix_encoder.parameters())
        
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
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
        
        print(f"\nTraining model for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            self.clip_model.train()
            self.classifier.train()
            if self.prefix_encoder is not None:
                self.prefix_encoder.train()
            
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
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * train_correct / train_total:.2f}%"
                })
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            self.clip_model.eval()
            self.classifier.eval()
            if self.prefix_encoder is not None:
                self.prefix_encoder.eval()
            
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
            
            # Update scheduler
            scheduler.step()
            
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
                if self.prefix_encoder is not None:
                    best_state_dict['prefix_encoder'] = self.prefix_encoder.state_dict()
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
            if self.prefix_encoder is not None and 'prefix_encoder' in best_state_dict:
                self.prefix_encoder.load_state_dict(best_state_dict['prefix_encoder'])
            print(f"\nLoaded best model with validation accuracy: {best_val_acc:.2f}%")
        
        return history
    
    def evaluate(self, dataloader, split_name: str = "test") -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate the trained model"""
        print(f"\nEvaluating on {split_name} set...")
        
        self.clip_model.eval()
        self.classifier.eval()
        if self.prefix_encoder is not None:
            self.prefix_encoder.eval()
        
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
        save_dict = {
            'clip_model_state_dict': self.clip_model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'adapter_type': self.adapter_type,
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes
        }
        if self.prefix_encoder is not None:
            save_dict['prefix_encoder_state_dict'] = self.prefix_encoder.state_dict()
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")


def run_adapter_finetuning(adapter_type: str = "lora"):
    """
    Main function to run adapter fine-tuning
    
    Args:
        adapter_type: Type of adapter ("lora", "bitfit", "prefix")
    """
    # Initialize fine-tuner
    finetuner = AdapterFineTuner(adapter_type=adapter_type)
    
    # Get data loaders with CLIP preprocessing
    train_loader, val_loader, test_loader = get_flowers_dataloaders(
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
        finetuner.save_model(os.path.join(config.RESULTS_DIR, f"{adapter_type}_finetuning.pth"))
    
    # Prepare results
    results = {
        'train': {'accuracy': train_acc, 'predictions': train_preds, 'labels': train_labels},
        'val': {'accuracy': val_acc, 'predictions': val_preds, 'labels': val_labels},
        'test': {'accuracy': test_acc, 'predictions': test_preds, 'labels': test_labels},
        'history': history
    }
    
    # Print summary
    print("\n" + "="*50)
    print(f"{adapter_type.upper()} ADAPTER FINE-TUNING RESULTS")
    print("="*50)
    print(f"Train Accuracy:      {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy:       {test_acc:.2f}%")
    print("="*50)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=str, default='lora', 
                       choices=['lora', 'bitfit', 'prefix'],
                       help='Adapter type to use')
    args = parser.parse_args()
    
    results = run_adapter_finetuning(adapter_type=args.adapter)
