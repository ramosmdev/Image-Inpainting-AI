# Architecture Deep Dive

This document outlines the technical architecture of the Image Inpainting AI, explaining the Neural Networks, the Data Loading strategy, and the Loss mechanisms.

---

## 1. The Generator — UNet (v2, 4-Level)

The core architecture defined in `src/models/unet.py` is a **deep U-Net**. A U-Net is an encoder-decoder network with skip connections, ideal for tasks where the output resolution must perfectly match the input resolution.

### How it processes data:
- **Input Concatenation**: The network receives an image and its corresponding binary mask. Instead of processing them separately, we concatenate them along the channel dimension. A 3-channel RGB image + a 1-channel mask becomes a **4-channel input**.
- **Encoder Path**: The network compresses spatial dimensions while increasing feature channels via 4 separate `conv_blocks` (Conv2D → BatchNorm → ReLU × 2) followed by MaxPooling. Channels grow: **64 → 128 → 256 → 512 → 1024** (bottleneck).
- **Bottleneck**: At 16×16 spatial resolution (1/16th of the 256×256 input), the 1024-channel bottleneck forces the model to build a **wide global context** before filling any hole. This is the key upgrade from the previous 3-level design, which bottlenecked at 32×32.
- **Decoder Path**: Using `ConvTranspose2d`, the network scales feature maps back up. At each step it **concatenates** features from the corresponding Encoder step (skip connections). This prevents the loss of high-resolution spatial detail.
- **BatchNorm**: Added to every conv block. At the increased depth (4 levels) and resolution (256×256), BatchNorm stabilizes gradient flow and prevents activation explosion.
- **Output**: The final output is forced through a `Sigmoid` activation, bounding pixel predictions strictly between 0 and 1, matching the normalized image format.

### Channel progression (256×256 input):
```
Input:      4ch  @ 256×256   (3ch RGB + 1ch mask)
enc1:      64ch  @ 256×256
enc2:     128ch  @ 128×128
enc3:     256ch  @ 64×64
enc4:     512ch  @ 32×32
bottleneck: 1024ch @ 16×16
dec4:     512ch  @ 32×32
dec3:     256ch  @ 64×64
dec2:     128ch  @ 128×128
dec1:      64ch  @ 256×256
out:        3ch  @ 256×256   (RGB reconstruction)
```

---

## 2. The Discriminator — Conditional PatchGAN

The discriminator is defined in `src/models/discriminator.py`.

### Why "Conditional"?
A standard discriminator sees only the composed image and must guess "real or fake?" But since only a brush-stroke region differs between real and fake images, a naive discriminator converges to random guessing (50/50), providing no useful gradient.

The **Conditional PatchGAN** solves this by concatenating the mask as a 4th input channel: `cat([image, mask], dim=1)`. The discriminator now explicitly knows *which region was generated* and can scrutinize it specifically — the same trick used in pix2pix for image-to-image translation.

### Architecture (256×256 input):
Four convolutional blocks with `InstanceNorm` and `LeakyReLU`, striding from 256×256 down to a patch validity map:
```
Input:   4ch  @ 256×256   (3ch image + 1ch mask)
Block 1: 64ch  @ 128×128  (no norm, first layer)
Block 2: 128ch @ 64×64
Block 3: 256ch @ 32×32
Block 4: 512ch @ 31×31
Output:  1ch   @ patch map (raw logits)
```
Outputs raw logits (no Sigmoid), consumed by `BCEWithLogitsLoss` for numerical stability.

---

## 3. Inpainting Dataset Pipeline

The `InpaintingDataset` (`src/datasets/inpainting_dataset.py`) wraps the COCO image dataset.

For every iteration, the dataset:
1. Opens the image and resizes it to `IMG_SIZE × IMG_SIZE` (currently **256×256**).
2. Dynamically generates an **Irregular Brush-Stroke Mask** using a random walk algorithm.
3. Applies the mask to zero out the hidden region.
4. Returns `masked_img`, `mask`, and `real_img` (ground truth).

### Irregular Mask Strategy
Rather than simple axis-aligned rectangles (easy for the model to exploit by learning a single square boundary), the mask generator (`src/utils/mask.py`) draws 1–3 random thick polylines using a random-walk algorithm:
- **Random starting point** anywhere in the image.
- **8–16 random steps** with displacement up to `W/6` per step.
- **Brush width** proportional to image size (based on `W/32` to `W/12`).

This produces organic, blob-like holes that force the model to generalize to arbitrary-shaped removal regions — matching real-world user behavior.

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
- VGG feature L1 magnitudes are 50–100× larger than pixel L1. Naively weighting this dominates and destabilizes training.
- **Progressive warmup**: `lambda_perceptual` starts at `0.0`, begins ramping at epoch `EPOCHS//3` (epoch 50 of 150), reaches `0.01` max at the final epoch. This lets the pixel loss establish structural correctness first.

### Total Generator Loss:
```
loss_G = context_loss + (0.3 × adv_loss) + (0.0→0.01 × perceptual_loss)
```
