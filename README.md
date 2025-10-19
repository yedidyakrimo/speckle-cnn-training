# Speckle CNN Training with GPU Optimization

A project for training CNN models to recognize speckle patterns with advanced GPU optimizations.

## Installation

### Automatic Installation (Recommended)
```bash
python setup.py
```

### Manual Installation
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Check GPU availability:
```bash
python check_gpu.py
```

3. Ensure you have CUDA installed on your computer (for GPU usage).

### System Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support (for GPU usage)
- 16+ GB RAM (recommended 32+ GB)
- 10+ GB free disk space

## Performance Tips
For detailed performance optimization guide, see [PERFORMANCE_TIPS.md](PERFORMANCE_TIPS.md)

## Configuration

The `config.json` file contains all settings:

### GPU Settings
- `use_gpu`: Enable/disable GPU usage
- `gpu_id`: GPU identifier (0, 1, etc.)
- `mixed_precision`: Use mixed precision training
- `compile_model`: Compile model for better performance
- `memory_fraction`: Fraction of available memory to use (0.9 = 90%)
- `allow_growth`: Dynamic GPU memory growth

### Training Settings
- `batch_size`: Batch size (16)
- `epochs`: Number of epochs (20)
- `learning_rate`: Learning rate (0.001)
- `optimizer`: Optimizer (AdamW)
- `num_workers`: Number of worker processes for data loading (8)
- `gradient_accumulation_steps`: Gradient accumulation steps (2)
- `weight_decay`: Weight decay (0.01)

### Performance Settings
- `pin_memory`: Use pin memory
- `persistent_workers`: Keep workers between epochs
- `prefetch_factor`: Prefetch factor
- `gradient_clip_norm`: Gradient clipping
- `early_stopping_patience`: Early stopping patience
- `save_best_model`: Save best model

## Usage

### Basic Usage
```bash
python train_speckles_cnn.py
```

### Running with Different Configurations

#### Windows (Double-click)
- `run_gpu.bat` - Run with GPU
- `run_cpu.bat` - Run with CPU

#### Command Line
```bash
# Run with advanced GPU settings
python run_training.py --gpu

# Run with CPU settings
python run_training.py --cpu

# Run with specific config file
python run_training.py --config config_high_performance.json
```

### Available Configuration Files
- `config.json`: Basic settings with GPU
- `config_high_performance.json`: Advanced settings for powerful computers
- `config_cpu.json`: Settings for CPU-only training

### Usage Examples
```bash
# Run various examples
python examples.py

# Check GPU availability
python check_gpu.py
```

## Optimizations Added

1. **Mixed Precision Training**: Using AMP for faster performance
2. **DataLoader Optimization**: pin_memory, persistent_workers, prefetch_factor
3. **Gradient Accumulation**: Training with larger batches
4. **Learning Rate Scheduling**: CosineAnnealingLR or ReduceLROnPlateau
5. **Early Stopping**: Early stopping to prevent overfitting
6. **Model Compilation**: Model compilation (PyTorch 2.0+)
7. **Memory Management**: Advanced GPU memory management
8. **Best Model Saving**: Save model with lowest validation loss

## Outputs

- `predictions.csv`: Model predictions
- `speckle_cnn.pth`: Best model
- `normalized_confusion_matrix.png`: Normalized confusion matrix

## System Requirements

- GPU with CUDA support (recommended RTX 3090 Ti or similar)
- 16+ GB RAM
- Python 3.8+
- PyTorch with CUDA support
