# Architecture Deep Dive

This document outlines the technical architecture of the Image Inpainting AI, explaining the Neural Networks, the Data Loading strategy, and the Loss mechanisms.

---

## 1. The Generator — UNet

The core architecture defined in `src/models/unet.py` is a **U-Net**. A U-Net is an encoder-decoder network with skip connections, ideal for tasks where the output resolution must perfectly match the input resolution.

### How it processes data:
- **Input Concatenation**: The network receives an image and its corresponding binary mask. Instead of processing them separately, we concatenate them along the channel dimension. A 3-channel RGB image + a 1-channel mask becomes a **4-channel input**.
- **Encoder Path**: The network compresses the spatial dimensions while increasing feature channels via 3 separate `conv_blocks` (Conv2D → ReLU → Conv2D → ReLU) followed by MaxPooling. Channels grow: 64 → 128 → 256 → 512.
- **Decoder Path**: Using `ConvTranspose2d`, the network scales the feature maps back up. Crucially, at each step it **concatenates** the features from the corresponding Encoder step (skip connections). This prevents the loss of high-resolution spatial detail.
- **Output**: The network forces the final output through a `Sigmoid` activation function, bounding pixel predictions strictly between 0 and 1, matching the normalized image format.

---

## 2. The Discriminator — Conditional PatchGAN

The discriminator is defined in `src/models/discriminator.py`.

### Why "Conditional"?
A standard discriminator sees only the composed image and must guess "real or fake?" But since only a small patch (20–50px) differs between real and fake images, a naive discriminator converges to random guessing (50/50), providing no useful gradient. 

The **Conditional PatchGAN** solves this by concatenating the mask as a 4th input channel: `cat([image, mask], dim=1)`. The discriminator now explicitly knows *which region was generated* and can scrutinize it specifically — the same trick used in pix2pix for image-to-image translation.

### Architecture:
Four convolutional blocks with `InstanceNorm` and `LeakyReLU`, striding from 128×128 down to a patch validity map. Outputs raw logits (no Sigmoid), consumed by `BCEWithLogitsLoss` for numerical stability.

---

## 3. Inpainting Dataset Pipeline

The `InpaintingDataset` (`src/datasets/inpainting_dataset.py`) wraps the COCO image dataset.

For every iteration, the dataset:
1. Opens the image and resizes it to `IMG_SIZE × IMG_SIZE` (currently 128×128).
2. Dynamically generates a **Random Square Mask** of size 20–50px at a random location.
3. Multiplies the image by the mask to zero out the hidden region.
4. Returns `masked_img`, `mask`, and `real_img` (ground truth).

---

## 4. Three-Way Loss System

Loss functions are defined in `src/losses/loss.py`. The generator is trained with a weighted combination of three signals:

### 4a. Context Loss (Weighted L1)
```
loss = (hole_L1 × 6.0) + valid_L1
```
- `hole_L1`: average absolute error inside the missing region only (`1 - mask`).
- `valid_L1`: average absolute error on the visible context region (`mask`).
- The 6× hole weighting ensures gradients focus where the model must actually hallucinate.

### 4b. Adversarial Loss
```
g_adv_loss = BCEWithLogitsLoss(discriminator(comp_img, mask), ones)
```
- The Generator is rewarded for producing outputs the conditional Discriminator cannot distinguish from real.
- `lambda_adv = 0.3` — increased to give this signal real weight after the discriminator upgrade. 

### 4c. Perceptual Loss (VGG-16, Progressive Warmup)
```
g_perceptual_loss = L1(VGG_features(comp_img), VGG_features(real_img))
```
- Uses three intermediate VGG-16 feature blocks to compare semantic texture, not just pixel values.
- VGG feature L1 magnitudes are 50–100× larger than pixel L1. Naively weighting this dominates and destabilizes training (causes green artifact / mode collapse).
- **Progressive warmup**: `lambda_perceptual` starts at `0.0`, begins ramping at epoch `EPOCHS//3`, reaches `0.01` max at the final epoch. This lets the pixel loss establish structural correctness first.

### Total Generator Loss:
```
loss_G = context_loss + (0.3 × adv_loss) + (0.0→0.01 × perceptual_loss)
```

