"""
Script to compare different trained models

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Usage:
    python scripts/compare_models.py --checkpoints checkpoints/gen1.pth checkpoints/gen2.pth
"""

import torch
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dcgan_model import Generator
from utils import save_image_grid
import config


def generate_comparison(checkpoints, nz=100, ngf=64, nc=3, image_size=64, 
                       num_samples=64, output_path='model_comparison.png'):
    """
    Generate images from multiple models for comparison
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use same noise for all models
    noise = torch.randn(num_samples, nz, 1, 1, device=device)
    
    all_images = []
    
    for i, checkpoint_path in enumerate(checkpoints):
        print(f"Loading model {i+1}: {checkpoint_path}")
        generator = Generator(nz=nz, ngf=ngf, nc=nc, image_size=image_size).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['model_state_dict'])
        generator.eval()
        
        with torch.no_grad():
            images = generator(noise).detach().cpu()
            all_images.append(images)
    
    # Create comparison grid
    num_models = len(checkpoints)
    fig, axes = plt.subplots(num_models, 1, figsize=(12, 4 * num_models))
    
    if num_models == 1:
        axes = [axes]
    
    for i, images in enumerate(all_images):
        grid = make_grid(images[:16], nrow=4, normalize=True, value_range=(-1, 1))
        grid_np = grid.numpy().transpose((1, 2, 0))
        grid_np = np.clip(grid_np, 0, 1)
        
        axes[i].imshow(grid_np)
        axes[i].axis('off')
        axes[i].set_title(f'Model {i+1}: {os.path.basename(checkpoints[i])}', 
                         fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to {output_path}")
    plt.close()


def main():
    """
    Main function
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    parser = argparse.ArgumentParser(description='Compare trained models')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                       help='Paths to generator checkpoints')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples per model')
    parser.add_argument('--output', type=str, default='model_comparison.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Model Comparison Tool")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    generate_comparison(
        args.checkpoints,
        nz=config.NZ,
        ngf=config.NGF,
        nc=config.NC,
        image_size=config.IMAGE_SIZE,
        num_samples=args.num_samples,
        output_path=args.output
    )


if __name__ == '__main__':
    main()

