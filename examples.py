#!/usr/bin/env python3
"""
Examples of how to use the Speckle CNN training system
"""

import json
import os
from train_speckles_cnn import main, load_config, setup_gpu

def example_basic_training():
    """Example: Basic training with default configuration"""
    print("=== Example 1: Basic Training ===")
    print("Running training with default configuration...")
    main(config_path='config.json')

def example_custom_config():
    """Example: Training with custom configuration"""
    print("=== Example 2: Custom Configuration ===")
    
    # Load base configuration
    config = load_config('config.json')
    
    # Modify settings for faster training (for demo purposes)
    config['training']['epochs'] = 3
    config['training']['batch_size'] = 8
    config['data']['k_folds'] = 2
    
    # Save custom configuration
    with open('config_demo.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Running training with custom configuration...")
    main(config_path='config_demo.json')
    
    # Clean up
    os.remove('config_demo.json')

def example_gpu_setup():
    """Example: GPU setup and information"""
    print("=== Example 3: GPU Setup ===")
    
    config = load_config('config.json')
    device, scaler = setup_gpu(config)
    
    print(f"Device: {device}")
    print(f"Mixed precision scaler: {scaler is not None}")
    
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

def example_config_comparison():
    """Example: Compare different configurations"""
    print("=== Example 4: Configuration Comparison ===")
    
    configs = ['config.json', 'config_high_performance.json', 'config_cpu.json']
    
    for config_file in configs:
        if os.path.exists(config_file):
            print(f"\n{config_file}:")
            config = load_config(config_file)
            
            print(f"  Batch size: {config['training']['batch_size']}")
            print(f"  Epochs: {config['training']['epochs']}")
            print(f"  GPU enabled: {config['gpu']['use_gpu']}")
            print(f"  Mixed precision: {config['gpu']['mixed_precision']}")
            print(f"  Optimizer: {config['training']['optimizer']}")

def example_memory_optimization():
    """Example: Memory optimization settings"""
    print("=== Example 5: Memory Optimization ===")
    
    # Configuration for low-memory GPU
    low_memory_config = {
        "data": {
            "data_dir": "D:/sergag/AI code Sergey Agdarov/Videos/Lili/TNT/Amygdala",
            "block_size": 50,
            "frame_size": [224, 224],
            "test_size": 0.2,
            "k_folds": 3
        },
        "model": {
            "conv1_channels": 16,
            "conv2_channels": 32,
            "conv3_channels": 64,
            "kernel_size": 3,
            "padding": 1,
            "pool_size": 2,
            "fc_hidden_size": 128,
            "dropout": 0.5
        },
        "training": {
            "batch_size": 4,
            "epochs": 10,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": False,
            "gradient_accumulation_steps": 4,
            "weight_decay": 0.01,
            "scheduler": "CosineAnnealingLR",
            "scheduler_params": {
                "T_max": 10,
                "eta_min": 0.0001
            }
        },
        "gpu": {
            "use_gpu": True,
            "gpu_id": 0,
            "mixed_precision": True,
            "compile_model": False,
            "memory_fraction": 0.7,
            "allow_growth": True
        },
        "performance": {
            "prefetch_factor": 1,
            "cache_dataset": False,
            "use_amp": True,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 5,
            "save_best_model": True
        }
    }
    
    # Save low memory configuration
    with open('config_low_memory.json', 'w') as f:
        json.dump(low_memory_config, f, indent=2)
    
    print("Created config_low_memory.json for low-memory GPU")
    print("Key optimizations:")
    print("  - Small batch size (4)")
    print("  - Gradient accumulation (4 steps)")
    print("  - Reduced memory fraction (0.7)")
    print("  - Disabled model compilation")
    print("  - Reduced prefetch factor")

def main_examples():
    """Run all examples"""
    print("Speckle CNN Training Examples")
    print("=" * 40)
    
    try:
        # Example 1: Basic training
        # example_basic_training()
        
        # Example 2: Custom configuration
        # example_custom_config()
        
        # Example 3: GPU setup
        example_gpu_setup()
        
        # Example 4: Configuration comparison
        example_config_comparison()
        
        # Example 5: Memory optimization
        example_memory_optimization()
        
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    import torch
    main_examples()
