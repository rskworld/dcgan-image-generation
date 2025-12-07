# DCGAN for Image Generation

**Author:** RSK World  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277

Deep Convolutional Generative Adversarial Network (DCGAN) for generating realistic images using adversarial training with convolutional layers.

## Description

This project implements DCGAN (Deep Convolutional GAN) for generating realistic images. The architecture uses convolutional layers in both generator and discriminator networks. It includes batch normalization, LeakyReLU activations, and proper weight initialization for stable training. Perfect for learning GAN fundamentals and image generation.

## Features

### Core Features
- ✅ DCGAN architecture with convolutional generator and discriminator
- ✅ Adversarial training with stable techniques
- ✅ Batch normalization and LeakyReLU activations
- ✅ Proper weight initialization
- ✅ Realistic image generation
- ✅ Support for multiple datasets (Custom, CelebA, CIFAR-10, MNIST)
- ✅ Training visualization and progress tracking
- ✅ Model checkpointing

### Advanced Features
- ✅ **Evaluation Metrics**: FID (Fréchet Inception Distance) and IS (Inception Score)
- ✅ **TensorBoard Integration**: Real-time training visualization
- ✅ **Latent Space Interpolation**: Generate smooth transitions between images
- ✅ **Advanced Training**: Learning rate scheduling, gradient clipping, label smoothing
- ✅ **Data Augmentation**: Multiple augmentation strategies including adaptive augmentation
- ✅ **Resume Training**: Continue training from checkpoints
- ✅ **Web Interface**: Interactive Flask web app for image generation
- ✅ **Early Stopping**: Automatic training termination
- ✅ **Model Comparison**: Compare different trained models

## Technologies

- Python
- PyTorch
- TensorFlow (optional)
- DCGAN
- Generative Adversarial Networks
- Convolutional Layers
- Image Generation
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rskworld/dcgan-image-generation.git
cd dcgan-image-generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using Jupyter Notebook (Recommended)

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `dcgan_training.ipynb` and run all cells.

### Option 2: Using Python Script

1. Prepare your dataset:
   - Place your images in `./data/custom/` directory
   - Or use built-in datasets (CelebA, CIFAR-10, MNIST)

2. Configure settings in `config.py`:
   - Set `DATASET_NAME` to 'custom', 'celeba', 'cifar10', or 'mnist'
   - Adjust hyperparameters as needed

3. Run training:
```bash
python main.py
```

## Project Structure

```
dcgan-image-generation/
├── dcgan_model.py          # Generator and Discriminator models
├── trainer.py              # Training logic and trainer class
├── data_loader.py          # Data loading utilities
├── utils.py                # Utility functions
├── config.py               # Configuration settings
├── main.py                 # Main training script
├── dcgan_training.ipynb    # Jupyter notebook for training
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Dataset directory
│   └── custom/            # Custom dataset images
├── outputs/                # Generated images and plots
└── checkpoints/            # Saved model checkpoints
```

## Configuration

Key hyperparameters in `config.py`:

- `NZ = 100`: Size of input noise vector
- `NGF = 64`: Number of generator filters
- `NDF = 64`: Number of discriminator filters
- `NC = 3`: Number of channels (3 for RGB, 1 for grayscale)
- `BATCH_SIZE = 128`: Training batch size
- `IMAGE_SIZE = 64`: Image size (64x64 or 128x128)
- `NUM_EPOCHS = 50`: Number of training epochs
- `LR = 0.0002`: Learning rate
- `BETA1 = 0.5`: Beta1 for Adam optimizer

## Training Tips

1. **Dataset**: Use high-quality images for better results. Recommended minimum: 1000+ images.

2. **Image Size**: Start with 64x64 images for faster training. Increase to 128x128 for better quality.

3. **Training Time**: Training can take several hours to days depending on:
   - Dataset size
   - Image resolution
   - Hardware (GPU recommended)

4. **Monitoring**: Check the generated samples in `outputs/` directory to monitor training progress.

5. **Stability**: If training becomes unstable:
   - Reduce learning rate
   - Adjust batch size
   - Check data normalization

## Advanced Usage

### Evaluation
```bash
python scripts/evaluate_model.py --checkpoint checkpoints/final_generator.pth
```

### Latent Interpolation
```bash
python scripts/interpolate_latent.py --checkpoint checkpoints/final_generator.pth
```

### TensorBoard
```bash
tensorboard --logdir runs/dcgan
```

### Web Interface
```bash
python web_app.py
```
Then open http://localhost:5000 in your browser.

See [FEATURES.md](FEATURES.md) for detailed documentation on all advanced features.

## Results

After training, you'll find:
- Generated images in `outputs/epoch_XXX_samples.png`
- Training loss plot in `outputs/training_losses.png`
- Model checkpoints in `checkpoints/`
- TensorBoard logs in `runs/dcgan/`

## License

This project is provided as-is for educational purposes.

## Contact

**RSK World**  
Website: https://rskworld.in  
Email: help@rskworld.in  
Phone: +91 93305 39277

## Acknowledgments

- Based on the DCGAN paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- PyTorch community for excellent documentation and examples

