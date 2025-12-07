"""
Visualize Generated Sample Data

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script visualizes the generated sample data.
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path


def visualize_sample_data(data_dir='data/custom', num_samples=16, save_path=None):
    """
    Visualize sample data
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        data_dir: Directory containing images
        num_samples: Number of samples to display
        save_path: Path to save visualization (None to display)
    """
    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(Path(data_dir).glob(ext))
    
    if not image_files:
        print(f"No images found in {data_dir}")
        return
    
    # Select random samples
    import random
    selected = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create grid
    cols = 4
    rows = (len(selected) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(selected):
        row = idx // cols
        col = idx % cols
        
        img = Image.open(img_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(img_path.name, fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(selected), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Sample Data from {data_dir} ({len(image_files)} total images)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main function
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize sample data')
    parser.add_argument('--data_dir', type=str, default='data/custom',
                       help='Directory containing images')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to display')
    parser.add_argument('--save', type=str, default='outputs/sample_data_preview.png',
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Sample Data Visualization")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    os.makedirs(os.path.dirname(args.save) if os.path.dirname(args.save) else '.', exist_ok=True)
    
    visualize_sample_data(args.data_dir, args.num_samples, args.save)


if __name__ == '__main__':
    main()

