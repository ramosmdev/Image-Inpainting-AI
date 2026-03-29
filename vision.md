# Project Vision and Roadmap

Our goal is to create a state-of-the-art Image Inpainting solution that remains accessible for individual developers to train and run on consumer-grade hardware like laptop GPUs.

---

## Phase 1: Improving Generation Quality

1. **Irregular Masks**: Transition from generating random square boxes to generating random, irregular "brush stroke" masks. Real-world users don't mask perfectly square regions; they brush over objects they want to eliminate.

2. **✅ Completed: Advanced GAN Architecture (Conditional PatchGAN)**: We successfully replaced the blurry L1 loss system with a state-of-the-art 3-way balance:
    - **Context Loss (Weighted L1)**: Anchors the underlying structure. The hole region is penalized 6× more than the visible area to focus gradients where learning is needed.
    - **Adversarial Loss (Conditional PatchGAN)**: A mask-conditioned Discriminator forces the UNet Generator to hallucinate photo-realistic, sharp patches. The discriminator sees both the image and the mask (4 channels), so it scrutinizes *specifically the inpainted region* rather than the full image.
    - **Perceptual Loss (VGG-16, Progressive Warmup)**: Extracts semantic feature maps to guarantee sharp textures match the context. Introduced with a warmup schedule — starts at epoch `EPOCHS//3` and ramps linearly — to prevent dominating the early structural learning phase.

3. **✅ Completed: Loss Weight Calibration**: Fixed a critical bug where `lambda_perceptual = 0.05` scaled a VGG loss magnitude of 50–100, making it 10× larger than the pixel loss and causing a green artifact / mode collapse. Correctly scaled to `0.01` max with progressive warmup.

4. **✅ Completed: Training Quality Monitoring**: Average loss tracking per epoch, progressive phase logging (Phase 1: structure only, Phase 2: perceptual refinement), explicit `train()`/`eval()` mode toggling per epoch.

---

## Phase 2: Architectural Evolution

1. **Higher Resolution Training**: Currently training at 128×128. Moving to 256×256 or 512×512 would dramatically improve texture fidelity. Requires more VRAM and compute.

2. **Attention Mechanisms**: Implementing Self-Attention blocks or transitioning to a Vision Transformer (ViT) backbone to help the model understand global image context better than standard convolutions.

3. **Latent Diffusion**: As a long-term goal, we may transition the architecture to a Stable-Diffusion-style architecture for significantly higher fidelity generation.

---

## Phase 3: Usability and Deployment

1. **Web Interface (Gradio/Streamlit)**: Build a simple web frontend where users can upload an image, draw a mask over an object they want removed, and instantly receive the inpainted result.

2. **ONNX / TensorRT Export**: Export the PyTorch model to standard inference formats to deploy to mobile devices or web browsers directly.

