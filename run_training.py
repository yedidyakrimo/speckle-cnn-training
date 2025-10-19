#!/usr/bin/env python3
"""
Script to run training with different configurations
"""

import argparse
import os
import sys
from train_speckles_cnn import main

def main_cli():
    parser = argparse.ArgumentParser(description='Run speckle CNN training with different configurations')
    parser.add_argument('--config', type=str, default='config.json',
                       choices=['config.json', 'config_high_performance.json', 'config_cpu.json'],
                       help='Configuration file to use')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Override config if GPU/CPU flags are used
    if args.gpu:
        args.config = 'config_high_performance.json'
        print("Using high-performance GPU configuration")
    elif args.cpu:
        args.config = 'config_cpu.json'
        print("Using CPU-only configuration")
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found!")
        print("Available configurations:")
        for config in ['config.json', 'config_high_performance.json', 'config_cpu.json']:
            if os.path.exists(config):
                print(f"  - {config}")
        sys.exit(1)
    
    print(f"Using configuration: {args.config}")
    print("=" * 50)
    
    # Run training
    try:
        main(config_path=args.config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_cli()
