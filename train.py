#!/usr/bin/env python3
"""
Training script for Transformer from scratch
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from encoder import Encoder
from decoder import Decoder
from utils import create_padding_mask, create_combined_mask, TranslationDataset, create_dataloader
from typing import Optional
import time
import matplotlib.pyplot as plt

class Transformer(nn.Module):
    """Complete Transformer model combining encoder and decoder"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, 
                 num_layers: int = 6, num_heads: int = 8, ff_dim: int = 2048, 
                 max_seq_len: int = 2048, dropout: float = 0.1, 
                 pos_encoding: str = 'rope', pad_idx: int = 0):
        """
        Args:
            src_vocab_size: Source vocabulary size (Finnish)
            tgt_vocab_size: Target vocabulary size (English)
            d_model: Model dimension
            num_layers: Number of encoder/decoder layers
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            pos_encoding: Positional encoding method ('rope' or 'relative_bias')
            pad_idx: Padding token index
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pos_encoding = pos_encoding
        self.pad_idx = pad_idx
        
        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pos_encoding=pos_encoding,
            pad_idx=pad_idx
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pos_encoding=pos_encoding,
            pad_idx=pad_idx
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize encoder and decoder weights
        for module in [self.encoder, self.decoder]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.constant_(p, 0.0)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_seq_len)
            tgt: Target sequence tensor of shape (batch_size, tgt_seq_len)
        
        Returns:
            Logits tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks
        src_mask = create_padding_mask(src, self.pad_idx)
        tgt_mask = create_combined_mask(tgt, self.pad_idx)
        
        # Encode source sequence
        encoder_output = self.encoder(src, src_mask)
        
        # Decode target sequence with encoder output
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        return decoder_output
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode source sequence (for inference)
        
        Args:
            src: Source sequence tensor of shape (batch_size, src_seq_len)
        
        Returns:
            Encoded sequence tensor of shape (batch_size, src_seq_len, d_model)
        """
        src_mask = create_padding_mask(src, self.pad_idx)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, 
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode target sequence (for inference)
        
        Args:
            tgt: Target sequence tensor of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output tensor
            tgt_mask: Optional target mask tensor
        
        Returns:
            Logits tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        if tgt_mask is None:
            tgt_mask = create_combined_mask(tgt, self.pad_idx)
        
        return self.decoder(tgt, encoder_output, tgt_mask)
    
    def count_parameters(self) -> dict:
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': encoder_params,
            'decoder': decoder_params
        }
    
    def print_model_summary(self):
        """Print model summary"""
        param_counts = self.count_parameters()
        
        print("=" * 60)
        print("TRANSFORMER MODEL SUMMARY")
        print("=" * 60)
        print(f"Source vocabulary size: {self.src_vocab_size:,}")
        print(f"Target vocabulary size: {self.tgt_vocab_size:,}")
        print(f"Model dimension (d_model): {self.d_model}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Number of attention heads: {self.num_heads}")
        print(f"Feedforward dimension: {self.ff_dim:,}")
        print(f"Maximum sequence length: {self.max_seq_len}")
        print(f"Dropout rate: {self.dropout}")
        print(f"Positional encoding: {self.pos_encoding}")
        print("-" * 60)
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")
        print(f"Encoder parameters: {param_counts['encoder']:,}")
        print(f"Decoder parameters: {param_counts['decoder']:,}")
        print("=" * 60)

class WarmupLR:
    """Learning rate scheduler with warmup and inverse sqrt decay"""
    
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        """
        Args:
            optimizer: PyTorch optimizer
            d_model: Model dimension
            warmup_steps: Number of warmup steps
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.d_model ** (-0.5) * (self.step_count / self.warmup_steps)
        else:
            # Inverse sqrt decay
            lr = self.d_model ** (-0.5) * (self.step_count ** (-0.5))
        
        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

class Trainer:
    """Training class for Transformer model"""
    
    def __init__(self, model: Transformer, config: argparse.Namespace, device: torch.device):
        """
        Args:
            model: Transformer model
            config: Training configuration
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = WarmupLR(
            optimizer=self.optimizer,
            d_model=config.d_model,
            warmup_steps=config.warmup_steps
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.gradient_norms = []
        self.parameter_stats = []
    
    def train_step(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple:
        """
        Single training step
        
        Args:
            src: Source sequence tensor
            tgt: Target sequence tensor
        
        Returns:
            Tuple of (loss, learning_rate, gradient_norm)
        """
        # Move tensors to device
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(src, tgt)
        
        # Prepare target for loss computation
        # Remove <bos> token and shift target for teacher forcing
        tgt_input = tgt[:, :-1]  # Remove last token
        tgt_output = tgt[:, 1:]  # Remove first token (<bos>)
        
        # Reshape logits and targets for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        targets_flat = tgt_output.contiguous().view(-1)
        
        # Compute loss
        loss = self.criterion(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate
        current_lr = self.scheduler.step()
        
        return loss.item(), current_lr, grad_norm.item()
    
    def validate(self, val_dataloader: DataLoader) -> float:
        """
        Validation step
        
        Args:
            val_dataloader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for src, tgt in val_dataloader:
                # Move tensors to device
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # Forward pass
                logits = self.model(src, tgt)
                
                # Prepare target for loss computation
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Reshape for loss computation
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                targets_flat = tgt_output.contiguous().view(-1)
                
                # Compute loss
                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1
        
        # Switch back to training mode
        self.model.train()
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def get_parameter_stats(self) -> dict:
        """Get parameter statistics for monitoring"""
        param_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_stats[f"{name}_mean"] = param.data.mean().item()
                param_stats[f"{name}_std"] = param.data.std().item()
                param_stats[f"{name}_grad_mean"] = param.grad.mean().item() if param.grad is not None else 0.0
                param_stats[f"{name}_grad_std"] = param.grad.std().item() if param.grad is not None else 0.0
        
        return param_stats
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.step_count,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'gradient_norms': self.gradient_norms,
            'parameter_stats': self.parameter_stats,
            'config': vars(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.model_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_path = os.path.join(self.config.model_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.step_count = checkpoint['scheduler_state_dict']
            self.best_val_loss = checkpoint['best_val_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.learning_rates = checkpoint['learning_rates']
            self.gradient_norms = checkpoint.get('gradient_norms', [])
            self.parameter_stats = checkpoint.get('parameter_stats', [])
            
            print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
            return checkpoint['epoch']
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            return 0
    
    def log_training_info(self, epoch: int, train_loss: float, val_loss: float, 
                          learning_rate: float, gradient_norm: float):
        """Log training information"""
        # Store metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(learning_rate)
        self.gradient_norms.append(gradient_norm)
        
        # Get parameter statistics
        param_stats = self.get_parameter_stats()
        self.parameter_stats.append(param_stats)
        
        # Print epoch summary
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {learning_rate:.6f} | "
              f"Grad Norm: {gradient_norm:.4f}")
        
        # Save training log
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'gradient_norm': gradient_norm,
            'parameter_stats': param_stats
        }
        
        log_path = os.path.join(self.config.model_dir, 'training_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_data)
        
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves"""
        if not self.train_losses:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate
        ax2.plot(epochs, self.learning_rates, 'g-', label='Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()
        ax2.grid(True)
        
        # Gradient norm
        ax3.plot(epochs, self.gradient_norms, 'm-', label='Gradient Norm')
        ax3.set_title('Gradient Norm')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Gradient Norm')
        ax3.legend()
        ax3.grid(True)
        
        # Parameter statistics (example: first layer weight mean)
        if self.parameter_stats:
            param_means = [stats.get('encoder.layers.0.self_attention.W_q.weight_mean', 0.0) 
                          for stats in self.parameter_stats]
            ax4.plot(epochs, param_means, 'c-', label='Q Weight Mean')
            ax4.set_title('Parameter Statistics (Q Weight Mean)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Weight Mean')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.model_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        
        # Close plot to free memory
        plt.close()
    
    def train_epoch(self, train_dataloader: DataLoader) -> tuple:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = len(train_dataloader)
        
        for batch_idx, (src, tgt) in enumerate(train_dataloader):
            # Training step
            loss, lr, grad_norm = self.train_step(src, tgt)
            
            total_loss += loss
            total_grad_norm += grad_norm
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1:4d}/{num_batches} | "
                      f"Loss: {loss:.4f} | LR: {lr:.6f}")
        
        avg_loss = total_loss / num_batches
        avg_grad_norm = total_grad_norm / num_batches
        
        return avg_loss, avg_grad_norm
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, 
              start_epoch: int = 0):
        """Main training loop"""
        print(f"\nStarting training from epoch {start_epoch + 1}")
        print("=" * 60)
        
        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)
            
            # Training
            train_loss, avg_grad_norm = self.train_epoch(train_dataloader)
            
            # Validation
            val_loss = self.validate(val_dataloader)
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log training info
            self.log_training_info(epoch + 1, train_loss, val_loss, current_lr, avg_grad_norm)
            
            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping check
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {self.config.patience} epochs without improvement")
                break
            
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch completed in {epoch_time:.2f} seconds")
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")

def get_config():
    """Get configuration from command line arguments or config file"""
    parser = argparse.ArgumentParser(description='Train Transformer from scratch')
    
    # Model architecture
    parser.add_argument('--src_vocab_size', type=int, default=5000, help='Source vocabulary size')
    parser.add_argument('--tgt_vocab_size', type=int, default=8000, help='Target vocabulary size')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pos_encoding', type=str, choices=['rope', 'relative_bias'], 
                       default='rope', help='Positional encoding method')
    
    # Training
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length for training')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Learning rate warmup steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Decoding strategy
    parser.add_argument('--decoding_strategy', type=str, choices=['greedy', 'beam', 'topk'], 
                       default='greedy', help='Decoding strategy')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models/', help='Model save directory')
    parser.add_argument('--config_file', type=str, help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    # Add pad_idx to config
    args.pad_idx = 0
    
    return args

def save_config(config, filepath: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(vars(config), f, indent=2)

def main():
    """Main training function"""
    # Get configuration
    config = get_config()
    
    # Create model directory
    os.makedirs(config.model_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.model_dir, 'config.json')
    save_config(config, config_path)
    print(f"Configuration saved to {config_path}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = Transformer(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        pos_encoding=config.pos_encoding,
        pad_idx=config.pad_idx
    )
    
    # Print model summary
    model.print_model_summary()
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Load checkpoint if specified
    start_epoch = 0
    if config.checkpoint:
        start_epoch = trainer.load_checkpoint(config.checkpoint)
    
    # TODO: Implement data loading and start training
    print("\nData loading not yet implemented. Training loop is ready!")
    print("To start training, implement data loading and call trainer.train()")
    
    # Save model
    model_path = os.path.join(config.model_dir, 'transformer_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
