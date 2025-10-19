#!/usr/bin/env python3
"""
Script to check GPU availability and configuration
"""

import torch
import sys

def check_gpu():
    print("=== GPU Configuration Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Test GPU memory allocation
        try:
            device = torch.device('cuda:0')
            test_tensor = torch.randn(1000, 1000).to(device)
            print(f"\nGPU Memory Test:")
            print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
            print("  GPU memory test: PASSED")
        except Exception as e:
            print(f"  GPU memory test: FAILED - {e}")
    else:
        print("No CUDA-capable GPU found. Training will use CPU.")
        print("For GPU training, ensure you have:")
        print("1. NVIDIA GPU with CUDA support")
        print("2. CUDA toolkit installed")
        print("3. PyTorch with CUDA support installed")
    
    print("\n=== Recommended Configuration ===")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 20:
            print("High-end GPU detected. Use 'config_high_performance.json'")
            print("Recommended settings:")
            print("  - batch_size: 32")
            print("  - num_workers: 12")
            print("  - mixed_precision: true")
        elif gpu_memory >= 8:
            print("Mid-range GPU detected. Use 'config.json'")
            print("Recommended settings:")
            print("  - batch_size: 16")
            print("  - num_workers: 8")
            print("  - mixed_precision: true")
        else:
            print("Low-memory GPU detected. Use 'config_cpu.json' or reduce batch_size")
    else:
        print("CPU-only mode. Use 'config_cpu.json'")
        print("Recommended settings:")
        print("  - batch_size: 4")
        print("  - num_workers: 2")
        print("  - gradient_accumulation_steps: 4")

if __name__ == "__main__":
    check_gpu()
