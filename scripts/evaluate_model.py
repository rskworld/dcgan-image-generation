"""
Script to evaluate trained DCGAN model using FID and IS metrics

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Usage:
    python scripts/evaluate_model.py --checkpoint checkpoints/final_generator.pth
"""

import torch
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dcgan_model import Generator
from data_loader import get_dataloader
from evaluation import evaluate_model
import config


def main():
    """
    Main evaluation function
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    parser = argparse.ArgumentParser(description='Evaluate DCGAN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to generator checkpoint')
    parser.add_argument('--num_samples', type=int, default=5000,
                       help='Number of samples for evaluation')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (overrides config)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("DCGAN Model Evaluation")
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
    
    # Load dataset
    dataset_name = args.dataset or config.DATASET_NAME
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        custom_dir=config.CUSTOM_DATA_DIR if dataset_name == 'custom' else None
    )
    
    # Evaluate
    results = evaluate_model(generator, dataloader, device, args.num_samples)
    
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)
    print(f"Inception Score: {results['inception_score']:.4f} ± {results['inception_score_std']:.4f}")
    print(f"FID Score: {results['fid_score']:.4f}")
    print("=" * 50)
    
    # Lower FID is better, higher IS is better
    print("\nNote: Lower FID is better, Higher IS is better")


if __name__ == '__main__':
    main()

