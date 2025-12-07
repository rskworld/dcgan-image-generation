"""
Script to generate latent space interpolations

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Usage:
    python scripts/interpolate_latent.py --checkpoint checkpoints/final_generator.pth
"""

import torch
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dcgan_model import Generator
from latent_interpolation import visualize_interpolation, generate_latent_walk
import config


def main():
    """
    Main function for latent interpolation
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    parser = argparse.ArgumentParser(description='Generate latent space interpolations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to generator checkpoint')
    parser.add_argument('--num_steps', type=int, default=10,
                       help='Number of interpolation steps')
    parser.add_argument('--interpolation', type=str, default='linear',
                       choices=['linear', 'spherical'],
                       help='Interpolation type')
    parser.add_argument('--output', type=str, default='outputs/interpolation.png',
                       help='Output file path')
    parser.add_argument('--walk', action='store_true',
                       help='Generate latent walk instead of interpolation')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Latent Space Interpolation")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator
    generator = Generator(
        nz=config.NZ,
        ngf=config.NGF,
        nc=config.NC,
        image_size=config.IMAGE_SIZE
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    print(f"Generator loaded from {args.checkpoint}")
    
    if args.walk:
        # Generate latent walk
        generate_latent_walk(
            generator,
            nz=config.NZ,
            num_images=args.num_steps,
            device=device,
            save_path=args.output
        )
    else:
        # Generate interpolation
        z1 = torch.randn(1, config.NZ, 1, 1)
        z2 = torch.randn(1, config.NZ, 1, 1)
        
        visualize_interpolation(
            generator,
            z1,
            z2,
            num_steps=args.num_steps,
            interpolation_type=args.interpolation,
            device=device,
            save_path=args.output
        )
    
    print(f"\nInterpolation saved to {args.output}")


if __name__ == '__main__':
    main()

