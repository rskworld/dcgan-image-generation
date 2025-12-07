"""
Generate Sample Data for DCGAN Training

Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script generates sample images that can be used for testing DCGAN training.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

# Try to import tqdm, fallback to simple progress if not available
try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    def tqdm(iterable, desc=""):
        """Simple progress indicator if tqdm is not available"""
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc} {i+1}/{total}", end='', flush=True)
            yield item
        print()  # New line after progress


def generate_simple_shapes(num_images=100, image_size=64, output_dir='data/custom'):
    """
    Generate simple geometric shapes as sample data
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        num_images: Number of images to generate
        image_size: Size of each image
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_images} simple shape images...")
    
    for i in tqdm(range(num_images)):
        # Create blank image
        img = Image.new('RGB', (image_size, image_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Random shape type
        shape_type = np.random.choice(['circle', 'rectangle', 'triangle', 'ellipse'])
        
        # Random colors
        fill_color = tuple(np.random.randint(0, 256, 3))
        outline_color = tuple(np.random.randint(0, 256, 3))
        
        # Random position and size
        size = np.random.randint(20, image_size // 2)
        x = np.random.randint(size, image_size - size)
        y = np.random.randint(size, image_size - size)
        
        if shape_type == 'circle':
            draw.ellipse([x-size, y-size, x+size, y+size], 
                        fill=fill_color, outline=outline_color, width=2)
        elif shape_type == 'rectangle':
            draw.rectangle([x-size, y-size, x+size, y+size], 
                          fill=fill_color, outline=outline_color, width=2)
        elif shape_type == 'ellipse':
            width = size
            height = np.random.randint(size//2, size*2)
            draw.ellipse([x-width, y-height, x+width, y+height], 
                        fill=fill_color, outline=outline_color, width=2)
        else:  # triangle
            points = [
                (x, y-size),
                (x-size, y+size),
                (x+size, y+size)
            ]
            draw.polygon(points, fill=fill_color, outline=outline_color, width=2)
        
        # Save image
        img.save(os.path.join(output_dir, f'shape_{i:04d}.png'))


def generate_gradient_images(num_images=100, image_size=64, output_dir='data/custom'):
    """
    Generate gradient images as sample data
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        num_images: Number of images to generate
        image_size: Size of each image
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_images} gradient images...")
    
    for i in tqdm(range(num_images)):
        # Create gradient
        img = Image.new('RGB', (image_size, image_size))
        pixels = img.load()
        
        # Random gradient direction
        direction = np.random.choice(['horizontal', 'vertical', 'diagonal', 'radial'])
        
        # Random colors
        color1 = np.random.randint(0, 256, 3)
        color2 = np.random.randint(0, 256, 3)
        
        for x in range(image_size):
            for y in range(image_size):
                if direction == 'horizontal':
                    t = x / image_size
                elif direction == 'vertical':
                    t = y / image_size
                elif direction == 'diagonal':
                    t = (x + y) / (2 * image_size)
                else:  # radial
                    center_x, center_y = image_size // 2, image_size // 2
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    t = dist / max_dist if max_dist > 0 else 0
                
                # Interpolate colors
                r = int(color1[0] * (1 - t) + color2[0] * t)
                g = int(color1[1] * (1 - t) + color2[1] * t)
                b = int(color1[2] * (1 - t) + color2[2] * t)
                
                pixels[x, y] = (r, g, b)
        
        # Save image
        img.save(os.path.join(output_dir, f'gradient_{i:04d}.png'))


def generate_pattern_images(num_images=100, image_size=64, output_dir='data/custom'):
    """
    Generate pattern images as sample data
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        num_images: Number of images to generate
        image_size: Size of each image
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_images} pattern images...")
    
    for i in tqdm(range(num_images)):
        # Create blank image
        img = Image.new('RGB', (image_size, image_size), color='white')
        draw = ImageDraw.Draw(img)
        
        # Random pattern type
        pattern_type = np.random.choice(['stripes', 'checkerboard', 'dots', 'waves'])
        
        # Random colors
        color1 = tuple(np.random.randint(0, 256, 3))
        color2 = tuple(np.random.randint(0, 256, 3))
        
        if pattern_type == 'stripes':
            stripe_width = np.random.randint(2, 8)
            for x in range(0, image_size, stripe_width * 2):
                draw.rectangle([x, 0, x + stripe_width, image_size], fill=color1)
                draw.rectangle([x + stripe_width, 0, x + stripe_width * 2, image_size], fill=color2)
        
        elif pattern_type == 'checkerboard':
            square_size = np.random.randint(4, 12)
            for x in range(0, image_size, square_size):
                for y in range(0, image_size, square_size):
                    if (x // square_size + y // square_size) % 2 == 0:
                        draw.rectangle([x, y, x + square_size, y + square_size], fill=color1)
                    else:
                        draw.rectangle([x, y, x + square_size, y + square_size], fill=color2)
        
        elif pattern_type == 'dots':
            dot_size = np.random.randint(2, 6)
            spacing = dot_size * 3
            for x in range(0, image_size, spacing):
                for y in range(0, image_size, spacing):
                    draw.ellipse([x, y, x + dot_size, y + dot_size], fill=color1)
        
        else:  # waves
            wave_freq = np.random.randint(2, 6)
            for y in range(image_size):
                wave = int(10 * np.sin(y * wave_freq / image_size * 2 * np.pi))
                x_center = image_size // 2 + wave
                draw.line([(x_center - 5, y), (x_center + 5, y)], fill=color1, width=2)
        
        # Save image
        img.save(os.path.join(output_dir, f'pattern_{i:04d}.png'))


def generate_mixed_data(num_images=500, image_size=64, output_dir='data/custom'):
    """
    Generate mixed sample data (shapes, gradients, patterns)
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    
    Args:
        num_images: Total number of images to generate
        image_size: Size of each image
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_images} mixed sample images...")
    print("This includes shapes, gradients, and patterns.")
    
    # Generate different types
    num_shapes = num_images // 3
    num_gradients = num_images // 3
    num_patterns = num_images - num_shapes - num_gradients
    
    print(f"\nGenerating {num_shapes} shape images...")
    generate_simple_shapes(num_shapes, image_size, output_dir)
    
    print(f"\nGenerating {num_gradients} gradient images...")
    generate_gradient_images(num_gradients, image_size, output_dir)
    
    print(f"\nGenerating {num_patterns} pattern images...")
    generate_pattern_images(num_patterns, image_size, output_dir)
    
    print(f"\n✓ Generated {num_images} sample images in {output_dir}")


def main():
    """
    Main function
    
    Author: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    """
    parser = argparse.ArgumentParser(description='Generate sample data for DCGAN training')
    parser.add_argument('--type', type=str, default='mixed',
                       choices=['shapes', 'gradients', 'patterns', 'mixed'],
                       help='Type of images to generate')
    parser.add_argument('--num_images', type=int, default=500,
                       help='Number of images to generate')
    parser.add_argument('--image_size', type=int, default=64,
                       help='Size of each image')
    parser.add_argument('--output_dir', type=str, default='data/custom',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Sample Data Generator")
    print("Author: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("=" * 50)
    
    if args.type == 'shapes':
        generate_simple_shapes(args.num_images, args.image_size, args.output_dir)
    elif args.type == 'gradients':
        generate_gradient_images(args.num_images, args.image_size, args.output_dir)
    elif args.type == 'patterns':
        generate_pattern_images(args.num_images, args.image_size, args.output_dir)
    else:  # mixed
        generate_mixed_data(args.num_images, args.image_size, args.output_dir)
    
    print("\n" + "=" * 50)
    print("Data generation completed!")
    print(f"Images saved to: {args.output_dir}")
    print("=" * 50)
    print("\nYou can now use this data for training:")
    print("  python main.py")
    print("  or")
    print("  jupyter notebook dcgan_training.ipynb")


if __name__ == '__main__':
    main()

