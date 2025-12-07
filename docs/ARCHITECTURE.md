# DCGAN Architecture Documentation

**Author:** RSK World  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277

## Overview

DCGAN (Deep Convolutional Generative Adversarial Network) uses convolutional layers in both the generator and discriminator networks, making it more stable and capable of generating higher quality images compared to the original GAN.

## Generator Architecture

The generator takes a random noise vector and transforms it into a realistic image through a series of transposed convolutions (also called deconvolutions).

### Architecture Details

For 64x64 images:
- Input: Random noise vector (100 dimensions)
- Layer 1: ConvTranspose2d(100 → 512 filters, 4x4, stride=1, padding=0)
- Layer 2: ConvTranspose2d(512 → 256 filters, 4x4, stride=2, padding=1)
- Layer 3: ConvTranspose2d(256 → 128 filters, 4x4, stride=2, padding=1)
- Layer 4: ConvTranspose2d(128 → 64 filters, 4x4, stride=2, padding=1)
- Layer 5: ConvTranspose2d(64 → 3 filters, 4x4, stride=2, padding=1)
- Output: 3-channel RGB image (64x64)

### Key Features
- Uses Batch Normalization after each layer (except input)
- Uses ReLU activation (except output layer which uses Tanh)
- No pooling layers
- No fully connected layers

## Discriminator Architecture

The discriminator takes an image and outputs a probability that it's real.

### Architecture Details

For 64x64 images:
- Input: 3-channel RGB image (64x64)
- Layer 1: Conv2d(3 → 64 filters, 4x4, stride=2, padding=1)
- Layer 2: Conv2d(64 → 128 filters, 4x4, stride=2, padding=1)
- Layer 3: Conv2d(128 → 256 filters, 4x4, stride=2, padding=1)
- Layer 4: Conv2d(256 → 512 filters, 4x4, stride=2, padding=1)
- Layer 5: Conv2d(512 → 1 filter, 4x4, stride=1, padding=0)
- Output: Single probability value

### Key Features
- Uses Batch Normalization after each layer (except input)
- Uses LeakyReLU activation (slope=0.2)
- No pooling layers
- No fully connected layers

## Training Process

1. **Discriminator Training**:
   - Train on real images (label=1)
   - Train on fake images from generator (label=0)
   - Update discriminator weights

2. **Generator Training**:
   - Generate fake images
   - Try to fool discriminator (label=1 for fake images)
   - Update generator weights

3. **Loss Functions**:
   - Binary Cross-Entropy Loss for both networks
   - Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
   - Generator: maximize log(D(G(z)))

## Best Practices

1. **Weight Initialization**: Normal distribution (mean=0, std=0.02)
2. **Batch Normalization**: Applied to all layers except input/output
3. **Learning Rate**: 0.0002 for Adam optimizer
4. **Beta1**: 0.5 for Adam optimizer
5. **Image Normalization**: Normalize to [-1, 1] range

## References

- Original DCGAN Paper: "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- PyTorch DCGAN Tutorial

