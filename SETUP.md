# Setup Guide - DCGAN Image Generation

**Author:** RSK World  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277

## Quick Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

#### Option A: Custom Dataset
1. Create a directory: `./data/custom/`
2. Place your training images in this directory
3. Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`
4. Recommended: At least 1000+ images for good results

#### Option B: Built-in Datasets
The project supports:
- **CelebA**: Automatically downloads when used
- **CIFAR-10**: Automatically downloads when used
- **MNIST**: Automatically downloads when used

### 3. Configure Settings

Edit `config.py` to adjust:
- Dataset name (`DATASET_NAME`)
- Image size (`IMAGE_SIZE`)
- Batch size (`BATCH_SIZE`)
- Number of epochs (`NUM_EPOCHS`)
- Learning rate (`LR`)

### 4. Start Training

#### Using Jupyter Notebook (Recommended):
```bash
jupyter notebook dcgan_training.ipynb
```

#### Using Python Script:
```bash
python main.py
```

## Directory Structure

After setup, your project should look like:

```
dcgan-image-generation/
├── dcgan_model.py
├── trainer.py
├── data_loader.py
├── utils.py
├── config.py
├── main.py
├── generate_samples.py
├── dcgan_training.ipynb
├── requirements.txt
├── README.md
├── SETUP.md
├── index.html
├── .gitignore
├── data/
│   └── custom/          # Your custom images go here
├── outputs/             # Generated images (created automatically)
└── checkpoints/         # Saved models (created automatically)
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `IMAGE_SIZE` (try 64 instead of 128)

### Training is Slow
- Use GPU if available (CUDA)
- Reduce `IMAGE_SIZE`
- Reduce number of images in dataset

### Poor Quality Results
- Train for more epochs
- Use larger dataset
- Adjust learning rate
- Check data normalization

## Contact

For issues or questions:
- **Website:** https://rskworld.in
- **Email:** help@rskworld.in
- **Phone:** +91 93305 39277

