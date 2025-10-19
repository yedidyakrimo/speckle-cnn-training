import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import time
import warnings
warnings.filterwarnings('ignore')


# 1. Load configuration file
def load_config(config_path='config.json'):
    """
    Reads parameters from JSON file.
    Args:
        config_path: Path to configuration file.
    Returns:
        config: Dictionary with parameters.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


# 1.5. GPU setup and configuration
def setup_gpu(config):
    """
    Sets up GPU configuration and returns device and scaler.
    Args:
        config: Configuration dictionary.
    Returns:
        device: PyTorch device (cuda or cpu).
        scaler: GradScaler for mixed precision training.
    """
    if config['gpu']['use_gpu'] and torch.cuda.is_available():
        device = torch.device(f'cuda:{config["gpu"]["gpu_id"]}')
        
        # Set memory fraction if specified
        if 'memory_fraction' in config['gpu']:
            torch.cuda.set_per_process_memory_fraction(config['gpu']['memory_fraction'])
        
        # Set allow growth if specified
        if config['gpu']['allow_growth']:
            torch.backends.cudnn.benchmark = True
        
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB")
        
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Initialize scaler for mixed precision
    scaler = GradScaler() if config['gpu']['mixed_precision'] and device.type == 'cuda' else None
    
    return device, scaler


# 2. Extract blocks from video
def extract_blocks_from_video(video_path, block_size, frame_size):
    """
    Extracts frame blocks from video file, converting frames to grayscale.
    Visualizes the first frame of the first block with folder indication (false/true).
    Args:
        video_path: Path to video file.
        block_size: Number of frames in block (50).
        frame_size: Tuple (width, height).
    Returns:
        blocks: List of blocks (each block is an array of block_size frames).
        block_info: List with block information (folder, filename, block number).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video {video_path}")
        return [], []

    blocks = []
    current_block = []
    block_info = []
    block_number = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        current_block.append(frame)

        if len(current_block) == block_size:
            blocks.append(np.array(current_block))
            folder_name = os.path.basename(os.path.dirname(video_path))
            file_name = os.path.basename(video_path)
            block_info.append({'folder': folder_name, 'file': file_name, 'block': block_number})
            if len(blocks) == 1:
                plt.imshow(blocks[0][0], cmap='gray')
                plt.title(f"Sample frame from {folder_name}/{file_name}")
                plt.axis('off')
                plt.show()
                plt.pause(1)
            current_block = []
            block_number += 1

    cap.release()
    if len(current_block) == block_size:
        blocks.append(np.array(current_block))
        block_info.append({'folder': folder_name, 'file': file_name, 'block': block_number})
    return blocks, block_info


# 3. Load data
def load_data(data_dir, block_size, frame_size):
    """
    Loads videos from 'yes'/'no' folders and extracts blocks.
    Args:
        data_dir: Path to video folder.
        block_size: Number of frames in block.
        frame_size: Frame size.
    Returns:
        X: Array of blocks (block_size, height, width).
        y: Class labels (0 for 'no', 1 for 'yes').
        block_info: List with block information (folder, file, block number).
    """
    X, y, block_info = [], [], []
    for label, class_name in enumerate(['no', 'yes']):
        class_dir = os.path.join(data_dir, class_name)
        for video_file in sorted([f for f in os.listdir(class_dir) if f.endswith('.avi')]):
            video_path = os.path.join(class_dir, video_file)
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}")
                continue
            blocks, video_block_info = extract_blocks_from_video(video_path, block_size, tuple(frame_size))
            X.extend(blocks)
            y.extend([label] * len(blocks))
            block_info.extend(video_block_info)
    return np.array(X), np.array(y), block_info


# 4. Custom dataset
class SpeckleDataset(Dataset):
    """
    Dataset for speckle pattern blocks.
    """

    def __init__(self, blocks, labels):
        self.blocks = blocks
        self.labels = labels

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        block = self.blocks[idx].astype(np.float32) / 255.0
        label = self.labels[idx]
        return torch.tensor(block, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# 5. CNN Model
class SpeckleCNN(nn.Module):
    """
    More complex CNN with three convolutional layers, BatchNorm and Dropout.
    """

    def __init__(self, block_size, frame_size, config):
        super(SpeckleCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, config['model']['conv1_channels'],
                               kernel_size=config['model']['kernel_size'],
                               padding=config['model']['padding'])
        self.bn1 = nn.BatchNorm3d(config['model']['conv1_channels'])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=config['model']['pool_size'])

        self.conv2 = nn.Conv3d(config['model']['conv1_channels'],
                               config['model']['conv2_channels'],
                               kernel_size=config['model']['kernel_size'],
                               padding=config['model']['padding'])
        self.bn2 = nn.BatchNorm3d(config['model']['conv2_channels'])
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=config['model']['pool_size'])

        self.conv3 = nn.Conv3d(config['model']['conv2_channels'],
                               config['model']['conv3_channels'],
                               kernel_size=config['model']['kernel_size'],
                               padding=config['model']['padding'])
        self.bn3 = nn.BatchNorm3d(config['model']['conv3_channels'])
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=config['model']['pool_size'])

        fc_input_size = (config['model']['conv3_channels'] *
                         (block_size // (config['model']['pool_size'] ** 3)) *
                         (frame_size[0] // (config['model']['pool_size'] ** 3)) *
                         (frame_size[1] // (config['model']['pool_size'] ** 3)))

        self.fc1 = nn.Linear(fc_input_size, config['model']['fc_hidden_size'])
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.fc2 = nn.Linear(config['model']['fc_hidden_size'], 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 6. Train and evaluate on one fold
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs, 
                      config, scheduler=None, scaler=None):
    """
    Trains the model and evaluates on validation set with advanced optimizations.
    Args:
        model: CNN model.
        train_loader, val_loader: Data loaders.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device (CPU/GPU).
        epochs: Number of epochs.
        config: Configuration dictionary.
        scheduler: Learning rate scheduler.
        scaler: GradScaler for mixed precision.
    Returns:
        val_accuracy: Accuracy on validation set.
        y_true, y_pred: True and predicted labels.
        best_val_loss: Best validation loss achieved.
    """
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    gradient_clip_norm = config['performance'].get('gradient_clip_norm', None)
    early_stopping_patience = config['performance'].get('early_stopping_patience', None)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, (blocks, labels) in enumerate(train_loader):
            blocks, labels = blocks.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision training
            if scaler is not None:
                with autocast():
                    outputs = model(blocks)
                    loss = criterion(outputs, labels)
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if gradient_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(blocks)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            running_loss += loss.item() * gradient_accumulation_steps
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            else:
                scheduler.step(running_loss / len(train_loader))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        y_pred, y_true = [], []
        
        with torch.no_grad():
            for blocks, labels in val_loader:
                blocks, labels = blocks.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if scaler is not None:
                    with autocast():
                        outputs = model(blocks)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(blocks)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(y_true, y_pred)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Early stopping
        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    
    return val_accuracy, y_true, y_pred, best_val_loss


# 7. Analyze predictions
def analyze_predictions(predictions_df, block_info):
    """
    Analyzes predictions.csv and outputs problematic blocks.
    Args:
        predictions_df: DataFrame with predictions.
        block_info: List with block information (folder, file, block number).
    """
    errors = predictions_df[predictions_df['Yes_Label'] != predictions_df['Predicted_Label']].copy()
    if errors.empty:
        print("No classification errors!")
        return

    print("\nProblematic blocks (classification errors):")
    for idx in errors.index:
        block_data = block_info[idx]
        true_label = 'yes' if errors.loc[idx, 'Yes_Label'] == 1 else 'no'
        pred_label = 'yes' if errors.loc[idx, 'Predicted_Label'] == 1 else 'no'
        print(f"Block {block_data['block']} in video {block_data['folder']}/{block_data['file']}: "
              f"True class = {true_label}, Predicted class = {pred_label}")


# 8. Main function with cross-validation
def main(config_path='config.json'):
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Read configuration
    config = load_config(config_path)
    
    # Setup GPU
    device, scaler = setup_gpu(config)

    # Check k_folds
    k_folds = config['data']['k_folds']
    if k_folds < 2:
        print(f"Error: k_folds must be >= 2, got {k_folds}. Setting k_folds=5.")
        k_folds = 5

    # Explicit paths for saving files
    output_dir = config['data']['data_dir']  # Update this path to your data directory
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    model_path = os.path.join(output_dir, 'speckle_cnn.pth')

    # Check number of frames
    total_frames = 2500  # Assume 2500 frames per video (50 blocks of 50 frames)
    blocks_per_video = total_frames // config['data']['block_size']
    print(f"Expected {blocks_per_video} blocks per video, total {blocks_per_video * 11} blocks for 11 videos")

    # Load data
    print("Loading data...")
    start_time = time.time()
    X, y, block_info = load_data(config['data']['data_dir'],
                                 config['data']['block_size'],
                                 config['data']['frame_size'])
    if len(X) == 0:
        print("Error: Failed to load data. Check file paths and formats.")
        return
    print(f"Loaded {len(X)} blocks, classes: {np.bincount(y)}")
    print(f"Data loading took {time.time() - start_time:.2f} seconds")

    # Cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    all_y_true, all_y_pred = [], []
    all_val_indices = []
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{k_folds}")
        fold_start_time = time.time()

        train_dataset = Subset(SpeckleDataset(X, y), train_idx)
        val_dataset = Subset(SpeckleDataset(X, y), val_idx)
        
        # Optimized DataLoader settings
        train_loader = DataLoader(train_dataset,
                                  batch_size=config['training']['batch_size'],
                                  shuffle=True,
                                  num_workers=config['training']['num_workers'],
                                  pin_memory=config['training'].get('pin_memory', False),
                                  persistent_workers=config['training'].get('persistent_workers', False),
                                  prefetch_factor=config['performance'].get('prefetch_factor', 2))
        val_loader = DataLoader(val_dataset,
                                batch_size=config['training']['batch_size'],
                                num_workers=config['training']['num_workers'],
                                pin_memory=config['training'].get('pin_memory', False),
                                persistent_workers=config['training'].get('persistent_workers', False),
                                prefetch_factor=config['performance'].get('prefetch_factor', 2))

        # Create model
        model = SpeckleCNN(config['data']['block_size'], config['data']['frame_size'], config).to(device)
        
        # Compile model if specified
        if config['gpu'].get('compile_model', False) and hasattr(torch, 'compile'):
            model = torch.compile(model)
            print("Model compiled for better performance")

        # Setup optimizer
        if config['training']['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), 
                                 lr=config['training']['learning_rate'],
                                 weight_decay=config['training'].get('weight_decay', 0))
        elif config['training']['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=config['training']['learning_rate'],
                                  weight_decay=config['training'].get('weight_decay', 0.01))
        else:
            optimizer = optim.SGD(model.parameters(), 
                                lr=config['training']['learning_rate'],
                                weight_decay=config['training'].get('weight_decay', 0))

        # Setup scheduler
        scheduler = None
        if config['training'].get('scheduler') == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, 
                                        T_max=config['training']['scheduler_params']['T_max'],
                                        eta_min=config['training']['scheduler_params']['eta_min'])
        elif config['training'].get('scheduler') == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        criterion = nn.CrossEntropyLoss()

        val_accuracy, y_true, y_pred, best_val_loss = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, device, 
            config['training']['epochs'], config, scheduler, scaler
        )
        
        fold_results.append(val_accuracy)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_val_indices.extend(val_idx)
        best_models.append((model.state_dict().copy(), best_val_loss))

        print(f"Accuracy on fold {fold + 1}: {val_accuracy:.4f}")
        print(f"Fold {fold + 1} took {time.time() - fold_start_time:.2f} seconds")
        
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Final metrics
    print("\nFinal cross-validation results:")
    print(f"Mean accuracy: {np.mean(fold_results):.4f} (±{np.std(fold_results):.4f})")
    print("Classification report:\n", classification_report(all_y_true, all_y_pred, target_names=['no', 'yes']))

    # Normalized Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['no', 'yes'],
                yticklabels=['no', 'yes'])
    plt.title('Normalized Confusion Matrix (Cross-Validation)')
    plt.xlabel('Predicted')
    plt.ylabel('Yes')
    plt.savefig(os.path.join(output_dir, 'normalized_confusion_matrix.png'))
    plt.show()

    # Save predictions
    predictions_df = pd.DataFrame({
        'Yes_Label': all_y_true,
        'Predicted_Label': all_y_pred,
        'Folder': [block_info[i]['folder'] for i in all_val_indices],
        'File': [block_info[i]['file'] for i in all_val_indices],
        'Block': [block_info[i]['block'] for i in all_val_indices]
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to '{predictions_path}'")

    # Analyze predictions
    analyze_predictions(predictions_df, block_info)

    # Save best model (lowest validation loss)
    if config['performance'].get('save_best_model', True):
        best_model_state, best_loss = min(best_models, key=lambda x: x[1])
        torch.save(best_model_state, model_path)
        print(f"Best model saved to '{model_path}' (validation loss: {best_loss:.4f})")
    else:
        # Save the last model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to '{model_path}'")
    
    # Print GPU memory usage
    if device.type == 'cuda':
        print(f"Final GPU Memory Usage: {torch.cuda.memory_allocated(device) / 1024**3:.1f} GB")
        print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device) / 1024**3:.1f} GB")


# Run
if __name__ == "__main__":
    main(config_path='config.json')