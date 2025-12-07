# Troubleshooting Guide

**Author:** RSK World  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277

## Common Issues and Solutions

### 1. CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `BATCH_SIZE` in `config.py` (try 32 or 64)
- Reduce `IMAGE_SIZE` (use 64 instead of 128)
- Close other applications using GPU
- Use CPU instead: Set `DEVICE = 'cpu'` in `config.py`

### 2. Training is Very Slow

**Problem**: Training takes too long

**Solutions**:
- Use GPU if available (check with `torch.cuda.is_available()`)
- Reduce `IMAGE_SIZE` to 64x64
- Reduce number of training images
- Reduce `NUM_EPOCHS` for testing
- Increase `BATCH_SIZE` if memory allows

### 3. Poor Quality Generated Images

**Problem**: Generated images look blurry or unrealistic

**Solutions**:
- Train for more epochs (increase `NUM_EPOCHS`)
- Use larger dataset (1000+ images recommended)
- Adjust learning rate (try 0.0001 or 0.0003)
- Check data normalization
- Ensure images are properly preprocessed
- Try different hyperparameters

### 4. Training Losses Not Converging

**Problem**: Losses oscillate or don't decrease

**Solutions**:
- Reduce learning rate
- Adjust batch size
- Check data quality
- Ensure proper weight initialization
- Try different optimizer parameters (beta1)

### 5. Discriminator Too Strong

**Problem**: Discriminator loss goes to 0, generator can't learn

**Solutions**:
- Reduce discriminator learning rate
- Train generator more often than discriminator
- Add noise to discriminator inputs
- Use label smoothing

### 6. Generator Produces Same Images

**Problem**: Generator produces identical or very similar images

**Solutions**:
- Increase noise vector size
- Add diversity loss
- Check for mode collapse
- Ensure proper randomization

### 7. Import Errors

**Problem**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version (3.7+ required)
- Verify you're in the correct directory
- Check PYTHONPATH

### 8. Dataset Not Found

**Problem**: `FileNotFoundError` or dataset loading errors

**Solutions**:
- For custom dataset: Create `data/custom/` and add images
- For built-in datasets: They will auto-download on first use
- Check dataset path in `config.py`
- Verify image file formats are supported

### 9. Checkpoint Loading Errors

**Problem**: Can't load saved checkpoints

**Solutions**:
- Verify checkpoint file exists
- Check model architecture matches checkpoint
- Ensure same hyperparameters used
- Check PyTorch version compatibility

### 10. Image Saving Issues

**Problem**: Generated images not saving or corrupted

**Solutions**:
- Check `outputs/` directory exists and is writable
- Verify matplotlib is installed
- Check disk space
- Ensure proper image format

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Review the configuration in `config.py`
3. Verify all dependencies are installed
4. Test with smaller dataset first
5. Check PyTorch and CUDA versions

**Contact**:
- **Website:** https://rskworld.in
- **Email:** help@rskworld.in
- **Phone:** +91 93305 39277

## Debugging Tips

1. **Start Small**: Test with small dataset and few epochs first
2. **Monitor Training**: Watch loss values and generated samples
3. **Check Data**: Visualize your training data to ensure it's correct
4. **Use Test Script**: Run `python test_setup.py` to verify setup
5. **Log Everything**: Save checkpoints regularly to track progress

