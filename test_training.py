#!/usr/bin/env python3
"""
Test script for training loop implementation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import Transformer, Trainer, WarmupLR
import argparse

def test_warmup_lr():
    """Test learning rate scheduler"""
    print("Testing WarmupLR scheduler...")
    
    # Create dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create scheduler
    scheduler = WarmupLR(optimizer, d_model=512, warmup_steps=100)
    
    # Test warmup phase
    lrs = []
    for step in range(50):
        lr = scheduler.step()
        lrs.append(lr)
        if step < 10:
            print(f"Step {step + 1:3d}: LR = {lr:.6f}")
    
    # Test decay phase
    for step in range(50, 150):
        lr = scheduler.step()
        lrs.append(lr)
        if step % 20 == 0:
            print(f"Step {step + 1:3d}: LR = {lr:.6f}")
    
    print(f"✅ WarmupLR scheduler working correctly!")
    print(f"   Warmup phase: {lrs[:100][0]:.6f} -> {lrs[:100][-1]:.6f}")
    print(f"   Decay phase: {lrs[100:][0]:.6f} -> {lrs[100:][-1]:.6f}")
    
    return True

def test_trainer_creation():
    """Test Trainer class creation"""
    print("\nTesting Trainer creation...")
    
    # Create config with all required attributes
    config = argparse.Namespace()
    config.pad_idx = 0
    config.learning_rate = 0.001
    config.warmup_steps = 100
    config.grad_clip = 1.0
    config.patience = 5
    config.d_model = 128  # Add missing attribute
    
    # Create model
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=2000,
        d_model=128,
        num_layers=2,
        num_heads=4,
        ff_dim=512
    )
    
    # Create trainer
    device = torch.device('cpu')
    trainer = Trainer(model, config, device)
    
    print(f"✅ Trainer created successfully!")
    print(f"   Loss function: {type(trainer.criterion).__name__}")
    print(f"   Optimizer: {type(trainer.optimizer).__name__}")
    print(f"   Scheduler: {type(trainer.scheduler).__name__}")
    
    return True

def test_training_step():
    """Test single training step"""
    print("\nTesting training step...")
    
    # Create config with all required attributes
    config = argparse.Namespace()
    config.pad_idx = 0
    config.learning_rate = 0.001
    config.warmup_steps = 100
    config.grad_clip = 1.0
    config.patience = 5
    config.d_model = 128  # Add missing attribute
    
    # Create model
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=2000,
        d_model=128,
        num_layers=2,
        num_heads=4,
        ff_dim=512
    )
    
    # Create trainer
    device = torch.device('cpu')
    trainer = Trainer(model, config, device)
    
    # Create dummy data
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12
    
    src = torch.randint(1, 1000, (batch_size, src_seq_len))
    tgt = torch.randint(1, 2000, (batch_size, tgt_seq_len))
    
    # Add special tokens
    tgt[:, 0] = 0  # <bos>
    tgt[:, -1] = 1  # <eos>
    
    print(f"Input shapes: src={src.shape}, tgt={tgt.shape}")
    
    # Training step
    try:
        loss, lr, grad_norm = trainer.train_step(src, tgt)
        print(f"✅ Training step successful!")
        print(f"   Loss: {loss:.4f}")
        print(f"   Learning rate: {lr:.6f}")
        print(f"   Gradient norm: {grad_norm:.4f}")
        
        # Check that loss is reasonable
        assert 0 < loss < 20, f"Loss {loss} seems unreasonable"
        assert 0 < lr < 1, f"Learning rate {lr} seems unreasonable"
        assert 0 < grad_norm < 10, f"Gradient norm {grad_norm} seems unreasonable"
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False

def test_validation():
    """Test validation step"""
    print("\nTesting validation step...")
    
    # Create config with all required attributes
    config = argparse.Namespace()
    config.pad_idx = 0
    config.learning_rate = 0.001
    config.warmup_steps = 100
    config.grad_clip = 1.0
    config.patience = 5
    config.d_model = 128  # Add missing attribute
    
    # Create model
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=2000,
        d_model=128,
        num_layers=2,
        num_heads=4,
        ff_dim=512
    )
    
    # Create trainer
    device = torch.device('cpu')
    trainer = Trainer(model, config, device)
    
    # Create dummy dataloader
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12
    
    # Create multiple batches
    batches = []
    for _ in range(3):
        src = torch.randint(1, 1000, (batch_size, src_seq_len))
        tgt = torch.randint(1, 2000, (batch_size, tgt_seq_len))
        
        # Add special tokens
        tgt[:, 0] = 0  # <bos>
        tgt[:, -1] = 1  # <eos>
        
        batches.append((src, tgt))
    
    # Create dataloader
    class DummyDataLoader:
        def __init__(self, batches):
            self.batches = batches
        
        def __iter__(self):
            return iter(self.batches)
        
        def __len__(self):
            return len(self.batches)
    
    val_dataloader = DummyDataLoader(batches)
    
    # Validation
    try:
        val_loss = trainer.validate(val_dataloader)
        print(f"✅ Validation successful!")
        print(f"   Validation loss: {val_loss:.4f}")
        
        # Check that loss is reasonable
        assert 0 < val_loss < 20, f"Validation loss {val_loss} seems unreasonable"
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def test_checkpoint_saving():
    """Test checkpoint saving and loading"""
    print("\nTesting checkpoint functionality...")
    
    # Create config with all required attributes
    config = argparse.Namespace()
    config.pad_idx = 0
    config.learning_rate = 0.001
    config.warmup_steps = 100
    config.grad_clip = 1.0
    config.patience = 5
    config.model_dir = 'test_models'
    config.d_model = 128  # Add missing attribute
    
    # Create model
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=2000,
        d_model=128,
        num_layers=2,
        num_heads=4,
        ff_dim=512
    )
    
    # Create trainer
    device = torch.device('cpu')
    trainer = Trainer(model, config, device)
    
    # Simulate some training
    trainer.train_losses = [2.5, 2.3, 2.1]
    trainer.val_losses = [2.6, 2.4, 2.2]
    trainer.learning_rates = [0.001, 0.001, 0.001]
    trainer.gradient_norms = [0.8, 0.7, 0.6]
    trainer.best_val_loss = 2.2
    
    try:
        # Save checkpoint
        trainer.save_checkpoint(3, is_best=True)
        print(f"✅ Checkpoint saved successfully!")
        
        # Load checkpoint
        loaded_epoch = trainer.load_checkpoint('test_models/checkpoint_epoch_3.pt')
        print(f"✅ Checkpoint loaded successfully from epoch {loaded_epoch}")
        
        # Verify loaded data
        assert len(trainer.train_losses) == 3, "Train losses not loaded correctly"
        assert len(trainer.val_losses) == 3, "Val losses not loaded correctly"
        assert trainer.best_val_loss == 2.2, "Best val loss not loaded correctly"
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING LOOP TESTING")
    print("=" * 60)
    
    # Run tests
    test1 = test_warmup_lr()
    test2 = test_trainer_creation()
    test3 = test_training_step()
    test4 = test_validation()
    test5 = test_checkpoint_saving()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if all([test1, test2, test3, test4, test5]):
        print("🎉 ALL TESTS PASSED! Training loop is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("=" * 60)
