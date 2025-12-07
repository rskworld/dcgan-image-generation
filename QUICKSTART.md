# Quick Start Guide

**Author:** RSK World  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277

Get started with DCGAN Image Generation in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Verify Setup

```bash
python test_setup.py
```

You should see: `✓ All tests passed! Setup is correct.`

## Step 3: Prepare Your Data

### Option A: Use Custom Dataset
1. Place your images in `data/custom/` directory
2. Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`
3. Recommended: 1000+ images

### Option B: Use Built-in Dataset
Edit `config.py` and set:
```python
DATASET_NAME = 'cifar10'  # or 'celeba', 'mnist'
```

## Step 4: Start Training

### Using Jupyter Notebook (Recommended)
```bash
jupyter notebook dcgan_training.ipynb
```
Then run all cells.

### Using Python Script
```bash
python main.py
```

## Step 5: Generate Images

After training, generate new images:
```bash
python generate_samples.py --checkpoint checkpoints/final_generator.pth
```

## Configuration

Edit `config.py` to adjust:
- `BATCH_SIZE`: 128 (reduce if out of memory)
- `IMAGE_SIZE`: 64 (64 or 128)
- `NUM_EPOCHS`: 50 (more for better results)
- `LR`: 0.0002 (learning rate)

## Troubleshooting

- **Out of memory?** Reduce `BATCH_SIZE` to 32 or 64
- **Training slow?** Use GPU or reduce `IMAGE_SIZE`
- **Poor quality?** Train more epochs or use larger dataset

See `docs/TROUBLESHOOTING.md` for more help.

## Next Steps

- Read `README.md` for detailed documentation
- Check `docs/ARCHITECTURE.md` for technical details
- Try `examples/example_usage.py` for code examples

## Contact

**RSK World**  
Website: https://rskworld.in  
Email: help@rskworld.in  
Phone: +91 93305 39277

