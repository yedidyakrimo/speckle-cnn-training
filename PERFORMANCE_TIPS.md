# Performance Tips - Speckle CNN Training

## GPU Optimizations

### 1. GPU Memory Settings
```json
"gpu": {
    "memory_fraction": 0.9,  // Use 90% of memory
    "allow_growth": true     // Dynamic memory growth
}
```

### 2. Mixed Precision Training
```json
"gpu": {
    "mixed_precision": true  // Enable mixed precision
}
```
- Saves ~50% memory
- 1.5-2x faster
- Requires GPU with Tensor Cores (RTX 20xx+)

### 3. Model Compilation (PyTorch 2.0+)
```json
"gpu": {
    "compile_model": true  // Compile the model
}
```
- 20-30% faster
- Requires PyTorch 2.0+

## DataLoader Optimizations

### 1. Worker Settings
```json
"training": {
    "num_workers": 8,           // Number of workers
    "pin_memory": true,         // Pin memory
    "persistent_workers": true  // Keep workers
}
```

### 2. Prefetch Factor
```json
"performance": {
    "prefetch_factor": 2  // Prefetch 2 batches
}
```

## Training Optimizations

### 1. Gradient Accumulation
```json
"training": {
    "gradient_accumulation_steps": 2  // Accumulate gradients
}
```
- Allows larger batch size
- Saves memory

### 2. Learning Rate Scheduling
```json
"training": {
    "scheduler": "CosineAnnealingLR",
    "scheduler_params": {
        "T_max": 20,
        "eta_min": 0.0001
    }
}
```

### 3. Early Stopping
```json
"performance": {
    "early_stopping_patience": 5  // Stop after 5 epochs without improvement
}
```

## Settings by GPU Type

### RTX 4090 / RTX 3090 Ti (24GB+)
```json
{
    "training": {
        "batch_size": 32,
        "num_workers": 12
    },
    "gpu": {
        "memory_fraction": 0.95,
        "mixed_precision": true,
        "compile_model": true
    }
}
```

### RTX 3080 / RTX 4070 (10-12GB)
```json
{
    "training": {
        "batch_size": 16,
        "num_workers": 8
    },
    "gpu": {
        "memory_fraction": 0.9,
        "mixed_precision": true,
        "compile_model": true
    }
}
```

### RTX 3060 / RTX 4060 (8GB)
```json
{
    "training": {
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "num_workers": 6
    },
    "gpu": {
        "memory_fraction": 0.8,
        "mixed_precision": true,
        "compile_model": false
    }
}
```

### Low Memory GPU (4-6GB)
```json
{
    "training": {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_workers": 4
    },
    "gpu": {
        "memory_fraction": 0.7,
        "mixed_precision": true,
        "compile_model": false
    }
}
```

## General Tips

### 1. Memory Monitoring
```python
# Check memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
```

### 2. Memory Cleanup
```python
# Clear memory between folds
torch.cuda.empty_cache()
```

### 3. System Settings
- Close other applications
- Use SSD for data storage
- Ensure sufficient RAM (16GB+)

### 4. Performance Testing
```bash
# Check GPU usage
nvidia-smi

# Run performance test
python check_gpu.py
```

## Troubleshooting Common Issues

### 1. Out of Memory
- Reduce batch_size
- Enable gradient_accumulation_steps
- Reduce memory_fraction
- Use mixed_precision

### 2. Slow Performance
- Increase num_workers
- Enable pin_memory
- Use persistent_workers
- Enable model compilation

### 3. CUDA Errors
- Check CUDA version
- Install PyTorch with CUDA support
- Check GPU compatibility

## Performance Measurement

### 1. Training Time
```python
import time
start_time = time.time()
# ... training code ...
print(f"Training took {time.time() - start_time:.2f} seconds")
```

### 2. Memory Usage
```python
import psutil
print(f"RAM usage: {psutil.virtual_memory().percent}%")
print(f"GPU usage: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
```

### 3. Throughput
```python
# Samples per second
samples_per_second = len(dataset) / training_time
print(f"Throughput: {samples_per_second:.2f} samples/sec")
```
