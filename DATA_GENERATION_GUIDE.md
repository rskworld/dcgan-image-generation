# Data Generation Guide

**Author:** RSK World  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277

This guide explains how to generate sample data for DCGAN training.

## Quick Start

Generate sample data with a single command:

```bash
python generate_sample_data.py --type mixed --num_images 500
```

## Data Types

### 1. Simple Shapes
Geometric shapes (circles, rectangles, triangles, ellipses) with random colors and positions.

```bash
python generate_sample_data.py --type shapes --num_images 200
```

### 2. Gradients
Color gradient images in various directions (horizontal, vertical, diagonal, radial).

```bash
python generate_sample_data.py --type gradients --num_images 200
```

### 3. Patterns
Pattern images including stripes, checkerboards, dots, and waves.

```bash
python generate_sample_data.py --type patterns --num_images 200
```

### 4. Mixed (Recommended)
Combination of all types for diverse training data.

```bash
python generate_sample_data.py --type mixed --num_images 500
```

## Command Line Options

```bash
python generate_sample_data.py [OPTIONS]

Options:
  --type TYPE          Type of images: shapes, gradients, patterns, or mixed
  --num_images N       Number of images to generate (default: 500)
  --image_size SIZE    Size of each image (default: 64)
  --output_dir DIR     Output directory (default: data/custom)
```

## Examples

### Generate 1000 images for training:
```bash
python generate_sample_data.py --type mixed --num_images 1000 --image_size 64
```

### Generate high-resolution test data:
```bash
python generate_sample_data.py --type mixed --num_images 200 --image_size 128
```

### Generate specific type:
```bash
python generate_sample_data.py --type shapes --num_images 300
```

## Visualize Generated Data

View a preview of generated images:

```bash
python visualize_sample_data.py --num_samples 16
```

This creates a grid visualization saved to `outputs/sample_data_preview.png`.

## Using Generated Data

After generating data, you can:

1. **Train DCGAN:**
   ```bash
   python main.py
   ```

2. **Use Jupyter Notebook:**
   ```bash
   jupyter notebook dcgan_training.ipynb
   ```

3. **Verify data loading:**
   ```python
   from data_loader import get_dataloader
   
   dataloader = get_dataloader(
       dataset_name='custom',
       custom_dir='data/custom',
       image_size=64,
       batch_size=128
   )
   ```

## Data Structure

Generated images are saved as:
```
data/custom/
├── shape_0000.png
├── shape_0001.png
├── gradient_0000.png
├── gradient_0001.png
├── pattern_0000.png
└── ...
```

## Tips

1. **More is Better**: Generate at least 500-1000 images for good results
2. **Consistent Size**: Use the same image_size as your training config
3. **Diversity**: Use 'mixed' type for more diverse training data
4. **Quality**: Higher resolution (128x128) takes longer but produces better results

## Custom Data

You can also add your own images to `data/custom/`:
- Supported formats: PNG, JPG, JPEG, BMP, GIF
- Images will be automatically resized during training
- Recommended: Square images work best

## Statistics

After generation, check:
- Total number of images: `ls data/custom | wc -l` (Linux/Mac) or `Get-ChildItem data\custom | Measure-Object | Select-Object Count` (Windows)
- File sizes: Ensure images are reasonable size
- Preview: Use `visualize_sample_data.py` to see samples

## Troubleshooting

**Problem**: No images generated
- Check output directory exists
- Verify write permissions
- Check for errors in console

**Problem**: Images look corrupted
- Regenerate with different parameters
- Check image_size is reasonable (64, 128, 256)

**Problem**: Too few images
- Increase `--num_images` parameter
- Generate multiple batches and combine

## Next Steps

1. Generate sample data: `python generate_sample_data.py --type mixed --num_images 500`
2. Visualize data: `python visualize_sample_data.py`
3. Start training: `python main.py`
4. Monitor progress in `outputs/` directory

## Contact

**RSK World**  
Website: https://rskworld.in  
Email: help@rskworld.in  
Phone: +91 93305 39277

