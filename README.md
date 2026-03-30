# Generative AI Image Inpainting

This repository contains the training, inference code, and documentation for an Image Inpainting AI, optimized to train on consumer hardware (specifically NVIDIA RTX 5070 Laptop GPU) using the COCO Dataset.

We have structured the documentation to provide both a quick start guide and deeper architectural insights.

## Documentation Directory

If you want to dive deep into how this project works and its future, please check our documentation files:

- 🏗️ **[Architecture](docs/architecture.md)**: Deep dive into the Neural Network (UNet), the custom loss function, and the data pipeline.
- 🏋️ **[Training Guide & Optimization](docs/training_guide.md)**: Details on the optimizations implemented to make the training run remarkably fast on modern GPUs (CuDNN benchmark, pin memory, async transfers).
- 🔭 **[Vision & Roadmap](vision.md)**: The next steps, future improvements, and the long-term vision for this AI project.

## Getting Started Quickly

1. **Environment Setup**: Ensure your `.venv` is active and you have PyTorch installed with GPU support. Install additional requirements from `requirements.txt`.
2. **Download Data**: Run `python download_data.py` to acquire the COCO dataset to the data directory.
3. **Training**: Execute `python train.py` to begin the training process. The model checkpoint will save locally (e.g., `checkpoint.pth`).
4. **Inference**: Use `python infer.py` to evaluate the saved model checkpoint and inspect the restored images.

## Project Structure Overivew

```text
Image-Inpainting-AI/
│
├── src/                        # Core AI modules
│   ├── models/                 # Neural networks (unet.py, discriminator.py)
│   ├── losses/                 # Loss functions (loss.py w/ VGG & LSGAN MSE)
│   ├── datasets/               # PyTorch dataset wrappers (inpainting_dataset.py)
│   ├── utils/                  # Helper scripts (mask.py, visualize.py)
│   └── config.py               # Global configuration
│
├── docs/                       # Comprehensive documentation
│   ├── architecture.md         # Network architecture details
│   ├── training_guide.md       # Hardware optimization and training guide
│   └── training_log.md         # Chronological record of training sessions
│
├── checkpoints/                # Saved model weights
├── outputs/                    # Generated result images
├── data/                       # Dataset directory (e.g. train images)
│
├── download_data.py            # Dataset acquisition script
├── train.py                    # Main script to train the inpainting model
├── infer.py                    # Script to evaluate model on test images
└── vision.md                   # Long-term vision and future roadmap
```
