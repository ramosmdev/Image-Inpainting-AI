# Image Inpainting AI

This repository contains the training and inference code for an Image Inpainting AI model, developed with PyTorch and optimized for the NVIDIA RTX 5070 Laptop GPU. The project leverages the COCO dataset to train the model to reconstruct missing or masked regions in images.

## Workplan and Objectives

The development of this project is structured around the following core goals:

1. **Training Pipeline**: Set up a robust, functional PyTorch training pipeline using the COCO dataset.
2. **Performance Optimization**: Ensure full compatibility and apply PyTorch performance best practices to maximize training efficiency on an NVIDIA RTX 5070 Laptop GPU.
3. **Inference & Visualization**: Create an easy-to-use inference script (`infer.py`) to test the model on masked images and visualize the inpainting results.
4. **Quality Refinement**: Continuously troubleshoot and refine the reconstruction quality (using custom losses and architectures) to ensure accurate, high-fidelity image restoration.

## Project Structure

```text
Image-Inpainting-AI/
│
├── models/                     # Neural network architectures
│   └── unet.py                 # Core UNet model implementation
│
├── losses/                     # Loss functions for network optimization
│   └── loss.py                 # Custom loss definitions for image reconstruction
│
├── datasets/                   # Data loaders and Dataset classes
│   └── inpainting_dataset.py   # Dataset wrapper for COCO images and masks
│
├── utils/                      # Helper scripts
│   ├── mask.py                 # Mask generation logic for input masking
│   └── visualize.py            # Result visualization utilities
│
├── config.py                   # Global configuration (IMG_SIZE, BATCH_SIZE, LR, etc.)
├── download_data.py            # Script to download and extract the dataset (e.g., COCO val2017)
├── train.py                    # Main script to train the inpainting model
└── infer.py                    # Script to evaluate model on test images and display results
```

## Getting Started

1. **Environment Setup**: Ensure your `.venv` is active and you have PyTorch installed with GPU support. Install additional requirements from `requirements.txt`.
2. **Download Data**: Run `python download_data.py` to acquire the COCO dataset to the data directory.
3. **Training**: Execute `python train.py` to begin the training process. The model checkpoint will save locally (e.g., `checkpoint.pth`).
4. **Inference**: Use `python infer.py` to evaluate the saved model checkpoint and inspect the restored images.
