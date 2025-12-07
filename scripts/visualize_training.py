"""
Script to visualize training progress from saved checkpoints

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Usage:
    python scripts/visualize_training.py --checkpoint_dir checkpoints/
"""

import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dcgan_model import Generator
from utils import save_image_grid
import config


def load_losses_from_checkpoints(checkpoint_dir):
    """
    Load training losses from all checkpoints
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    G_losses = []
    D_losses = []
    epochs = []
    
    checkpoint_files = sorted(Path(checkpoint_dir).glob('generator_epoch_*.pth'))
    
    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if 'losses' in checkpoint:
                G_losses.extend(checkpoint['losses'])
                epochs.append(checkpoint.get('epoch', 0))
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_file}: {e}")
    
    return G_losses, D_losses, epochs


def visualize_progress(checkpoint_dir, output_dir='outputs'):
    """
    Visualize training progress
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    print("Loading checkpoints...")
    G_losses, D_losses, epochs = load_losses_from_checkpoints(checkpoint_dir)
    
    if not G_losses:
        print("No loss data found in checkpoints")
        return
    
    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.plot(G_losses, label='Generator Loss', alpha=0.7)
    if D_losses:
        plt.plot(D_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'training_progress.png')
    plt.savefig(output_path, dpi=150)
    print(f"Progress plot saved to {output_path}")
    plt.close()


def main():
    """
    Main function
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    parser = argparse.ArgumentParser(description='Visualize training progress')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Training Progress Visualization")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    os.makedirs(args.output_dir, exist_ok=True)
    visualize_progress(args.checkpoint_dir, args.output_dir)


if __name__ == '__main__':
    main()

