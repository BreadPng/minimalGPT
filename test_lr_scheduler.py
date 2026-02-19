#!/usr/bin/env python3
"""
Test script to visualize the learning rate scheduler with warm-up and linear decay.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Mock configuration class
class MockConfig:
    def __init__(self):
        self.learning_rate = 1e-4
        self.warmup_iters = 1000
        self.lr_decay_iters = 8000
        self.min_lr = 1e-5

# Mock optimizer
class MockOptimizer:
    def __init__(self):
        self.param_groups = [{'lr': 1e-4}]

def get_lr_scheduler(optimizer, config):
    """
    Create a learning rate scheduler with warm-up and linear decay.
    
    Args:
        optimizer: The optimizer to schedule
        config: Training configuration containing lr schedule parameters
    
    Returns:
        A function that updates the learning rate based on current iteration
    """
    def update_lr(iter_num):
        # Warm-up phase: linear increase from 0 to target learning rate
        if iter_num < config.warmup_iters:
            lr = config.learning_rate * iter_num / config.warmup_iters
        # Decay phase: linear decrease from target learning rate to min_lr
        elif iter_num < config.warmup_iters + config.lr_decay_iters:
            decay_ratio = (iter_num - config.warmup_iters) / config.lr_decay_iters
            lr = config.learning_rate - (config.learning_rate - config.min_lr) * decay_ratio
        # Constant phase: maintain min_lr
        else:
            lr = config.min_lr
        
        # Update learning rate for all parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    return update_lr

def test_lr_scheduler():
    """Test and visualize the learning rate scheduler."""
    
    # Create mock objects
    config = MockConfig()
    optimizer = MockOptimizer()
    
    # Create scheduler
    lr_scheduler = get_lr_scheduler(optimizer, config)
    
    # Test parameters
    max_iters = 12000
    iterations = np.arange(max_iters)
    learning_rates = []
    
    # Calculate learning rates for all iterations
    for iter_num in iterations:
        lr = lr_scheduler(iter_num)
        learning_rates.append(lr)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot learning rate schedule
    plt.subplot(2, 1, 1)
    plt.plot(iterations, learning_rates, 'b-', linewidth=2)
    plt.axvline(x=config.warmup_iters, color='r', linestyle='--', alpha=0.7, label=f'Warm-up end ({config.warmup_iters})')
    plt.axvline(x=config.warmup_iters + config.lr_decay_iters, color='g', linestyle='--', alpha=0.7, label=f'Decay end ({config.warmup_iters + config.lr_decay_iters})')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule: Warm-up + Linear Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot learning rate in linear scale for better visibility of warm-up
    plt.subplot(2, 1, 2)
    plt.plot(iterations, learning_rates, 'b-', linewidth=2)
    plt.axvline(x=config.warmup_iters, color='r', linestyle='--', alpha=0.7, label=f'Warm-up end ({config.warmup_iters})')
    plt.axvline(x=config.warmup_iters + config.lr_decay_iters, color='g', linestyle='--', alpha=0.7, label=f'Decay end ({config.warmup_iters + config.lr_decay_iters})')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lr_schedule_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key statistics
    print("Learning Rate Schedule Statistics:")
    print(f"Initial LR: {learning_rates[0]:.2e}")
    print(f"Peak LR (after warm-up): {learning_rates[config.warmup_iters]:.2e}")
    print(f"Final LR: {learning_rates[-1]:.2e}")
    print(f"Warm-up iterations: {config.warmup_iters}")
    print(f"Decay iterations: {config.lr_decay_iters}")
    print(f"Total schedule length: {len(learning_rates)}")
    
    # Verify key points
    assert abs(learning_rates[0] - 0.0) < 1e-10, "Learning rate should start at 0"
    assert abs(learning_rates[config.warmup_iters] - config.learning_rate) < 1e-10, "Learning rate should reach target after warm-up"
    assert abs(learning_rates[-1] - config.min_lr) < 1e-10, "Learning rate should end at min_lr"
    
    print("\n✅ All tests passed! Learning rate scheduler is working correctly.")

if __name__ == "__main__":
    test_lr_scheduler()
