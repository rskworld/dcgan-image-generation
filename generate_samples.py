"""
Script to generate images using a trained DCGAN model

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Usage:
    python generate_samples.py --checkpoint checkpoints/final_generator.pth --num_samples 64
"""

import torch
import argparse
import os
from dcgan_model import Generator
from utils import save_image_grid
import config


def load_generator(checkpoint_path, device, nz=100, ngf=64, nc=3, image_size=64):
    """
    Load a trained generator from checkpoint
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        checkpoint_path: Path to generator checkpoint
        device: torch device
        nz: Size of noise vector
        ngf: Number of generator filters
        nc: Number of channels
        image_size: Image size
    
    Returns:
        Loaded generator model
    """
    generator = Generator(nz=nz, ngf=ngf, nc=nc, image_size=image_size).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    
    print(f"Generator loaded from {checkpoint_path}")
    return generator


def main():
    """
    Main function to generate samples
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    parser = argparse.ArgumentParser(description='Generate images using trained DCGAN')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/final_generator.pth',
                       help='Path to generator checkpoint')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='generated_samples.png',
                       help='Output file path')
    parser.add_argument('--nz', type=int, default=100,
                       help='Size of noise vector')
    parser.add_argument('--ngf', type=int, default=64,
                       help='Number of generator filters')
    parser.add_argument('--nc', type=int, default=3,
                       help='Number of channels')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Image size')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("DCGAN Image Generation")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    generator = load_generator(
        args.checkpoint,
        device,
        nz=args.nz,
        ngf=args.ngf,
        nc=args.nc,
        image_size=args.image_size
    )
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    with torch.no_grad():
        noise = torch.randn(args.num_samples, args.nz, 1, 1, device=device)
        fake_images = generator(noise).detach().cpu()
    
    # Save images
    save_image_grid(fake_images, args.output)
    print(f"Generated images saved to {args.output}")
    
    print("\nGeneration completed!")


if __name__ == '__main__':
    main()

