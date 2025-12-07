# DCGAN Image Generation - Project Information

**Author:** RSK World  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277

## Project Overview

This is a complete implementation of DCGAN (Deep Convolutional Generative Adversarial Network) for generating realistic images. The project includes:

- Full DCGAN architecture implementation
- Training scripts and utilities
- Jupyter notebook for interactive training
- Comprehensive documentation
- Demo HTML page

## Files Created

### Core Implementation Files
1. **dcgan_model.py** - Generator and Discriminator model definitions
2. **trainer.py** - Training logic and trainer class
3. **data_loader.py** - Data loading utilities for various datasets
4. **utils.py** - Utility functions for visualization and image saving
5. **config.py** - Configuration settings and hyperparameters

### Training Scripts
6. **main.py** - Main training script (command-line)
7. **dcgan_training.ipynb** - Jupyter notebook for interactive training
8. **generate_samples.py** - Script to generate images from trained models

### Documentation
9. **README.md** - Main project documentation
10. **SETUP.md** - Setup and installation guide
11. **PROJECT_INFO.md** - This file

### Configuration & Other Files
12. **requirements.txt** - Python dependencies
13. **.gitignore** - Git ignore patterns
14. **LICENSE** - MIT License
15. **index.html** - Demo HTML page
16. **test_setup.py** - Setup verification script

## Features Implemented

✅ DCGAN architecture with convolutional layers  
✅ Batch normalization and LeakyReLU activations  
✅ Proper weight initialization  
✅ Adversarial training loop  
✅ Support for multiple datasets (Custom, CelebA, CIFAR-10, MNIST)  
✅ Model checkpointing  
✅ Training visualization  
✅ Image generation utilities  
✅ Comprehensive documentation  

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py

# Train using notebook
jupyter notebook dcgan_training.ipynb

# Or train using script
python main.py
```

### Generate Images
```bash
python generate_samples.py --checkpoint checkpoints/final_generator.pth
```

## Project Structure

```
dcgan-image-generation/
├── dcgan_model.py          # Model definitions
├── trainer.py              # Training logic
├── data_loader.py          # Data utilities
├── utils.py                # Helper functions
├── config.py               # Configuration
├── main.py                 # Main script
├── generate_samples.py     # Image generation
├── test_setup.py           # Setup test
├── dcgan_training.ipynb    # Jupyter notebook
├── requirements.txt        # Dependencies
├── README.md               # Documentation
├── SETUP.md                # Setup guide
├── PROJECT_INFO.md         # This file
├── LICENSE                 # License
├── index.html              # Demo page
├── .gitignore              # Git ignore
├── data/                   # Dataset directory
├── outputs/                # Generated images
└── checkpoints/            # Saved models
```

## Contact Information

All files in this project include the following contact information in comments:

- **Author:** RSK World
- **Website:** https://rskworld.in
- **Email:** help@rskworld.in
- **Phone:** +91 93305 39277

## License

MIT License - See LICENSE file for details.

