"""
Test script to verify DCGAN setup

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Run this script to verify that all components are working correctly.
"""

import torch
import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        from dcgan_model import Generator, Discriminator, weights_init
        from trainer import DCGANTrainer
        from data_loader import get_dataloader
        from utils import save_image_grid, plot_training_losses
        import config
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_models():
    """Test if models can be created"""
    print("\nTesting model creation...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test Generator
        generator = Generator(nz=100, ngf=64, nc=3, image_size=64).to(device)
        print("✓ Generator created successfully")
        
        # Test Discriminator
        discriminator = Discriminator(nc=3, ndf=64, image_size=64).to(device)
        print("✓ Discriminator created successfully")
        
        # Test forward pass
        noise = torch.randn(4, 100, 1, 1, device=device)
        fake = generator(noise)
        print(f"✓ Generator forward pass: {fake.shape}")
        
        output = discriminator(fake)
        print(f"✓ Discriminator forward pass: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model test error: {e}")
        return False


def test_trainer():
    """Test if trainer can be created"""
    print("\nTesting trainer creation...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator = Generator(nz=100, ngf=64, nc=3, image_size=64).to(device)
        discriminator = Discriminator(nc=3, ndf=64, image_size=64).to(device)
        
        trainer = DCGANTrainer(
            generator=generator,
            discriminator=discriminator,
            device=device,
            lr=0.0002,
            beta1=0.5,
            nz=100
        )
        print("✓ Trainer created successfully")
        return True
    except Exception as e:
        print(f"✗ Trainer test error: {e}")
        return False


def test_directories():
    """Test if required directories exist or can be created"""
    print("\nTesting directories...")
    try:
        dirs = ['outputs', 'checkpoints', 'data']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
            if os.path.exists(dir_name):
                print(f"✓ Directory '{dir_name}' ready")
            else:
                print(f"✗ Failed to create '{dir_name}'")
                return False
        return True
    except Exception as e:
        print(f"✗ Directory test error: {e}")
        return False


def main():
    """
    Main test function
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    print("=" * 50)
    print("DCGAN Setup Test")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    results = []
    results.append(test_imports())
    results.append(test_models())
    results.append(test_trainer())
    results.append(test_directories())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All tests passed! Setup is correct.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

