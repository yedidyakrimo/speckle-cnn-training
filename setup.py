#!/usr/bin/env python3
"""
Setup script for Speckle CNN Training
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"  Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages"""
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing required packages"
    )

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        else:
            print("⚠ CUDA not available - will use CPU")
    except ImportError:
        print("⚠ PyTorch not installed - cannot check GPU")

def main():
    print("=== Speckle CNN Training Setup ===\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Setup failed during package installation")
        sys.exit(1)
    
    # Check GPU
    print("\nChecking GPU availability...")
    check_gpu()
    
    print("\n=== Setup Complete ===")
    print("You can now run:")
    print("  python check_gpu.py          # Check GPU configuration")
    print("  python run_training.py --gpu # Run with GPU")
    print("  python run_training.py --cpu # Run with CPU only")

if __name__ == "__main__":
    main()
