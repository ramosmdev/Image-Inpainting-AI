from src.datasets.inpainting_dataset import InpaintingDataset
from src.models.unet import UNet
from src.models.discriminator import Discriminator
from src.losses.loss import inpainting_loss, adversarial_loss, PerceptualLoss
import src.config as config
import torch
from torch.utils.data import DataLoader
import time
import os
import logging
import sys

# Configure logging to both console.log and terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("console.log", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # SPEED OPTIMIZATION 1: Optimizes convolution algorithms if input size doesn't change
        torch.backends.cudnn.benchmark = True 

    dataset = InpaintingDataset(config.DATA_PATH)
    # SPEED OPTIMIZATION 2: num_workers loads images in background, pin_memory speeds up CPU->GPU copy
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize BOTH networks
    generator = UNet().to(device)
    discriminator = Discriminator().to(device)
    
    # Setup Perceptual VGG Loss (Frozen weights)
    logger.info("Loading VGG-16 for Perceptual Loss...")
    perceptual_criterion = PerceptualLoss().to(device)
    perceptual_criterion.eval()
    
    # Dual Optimizers (TTUR: Equal learning rates for LSGAN to create a tougher critic!)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.LR, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.LR, betas=(0.5, 0.999))

    # Learning Rate Schedulers (Linear decay starting at epoch 100)
    def lr_lambda(epoch):
        decay_start = 100
        if epoch < decay_start:
            return 1.0
        else:
            return max(0.0, 1.0 - (epoch - decay_start) / (config.EPOCHS - decay_start))
            
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda)

    # --- Loss Weights ---
    # Adversarial loss (BCE logits) outputs ~0.3-1.0.
    # INCREASED from 0.1 -> 0.3: the discriminator now has context (mask-conditioned),
    # so its signal is more targetted. We need more weight to overcome blurry mean predictions.
    lambda_adv = 0.3
    # VGG feature L1 distances are HUGE (10-100+). Start at 0, ramp in after structure is learned.
    lambda_perceptual = 0.0
    # At what epoch to start introducing perceptual loss
    perceptual_warmup_epoch = config.EPOCHS // 3
    # Final perceptual weight after warmup
    lambda_perceptual_final = 0.01

    os.makedirs("checkpoints", exist_ok=True)

    logger.info(f"Starting GAN training with Perceptual Loss on {device}...")
    logger.info(f"Phase 1 (epochs 0-{perceptual_warmup_epoch-1}): Pixel + Adversarial loss only (building structure)")
    logger.info(f"Phase 2 (epochs {perceptual_warmup_epoch}+): Adding Perceptual loss to sharpen textures")
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        # Set both networks to training mode explicitly
        generator.train()
        discriminator.train()

        # Progressive perceptual warmup: ramps from 0 -> lambda_perceptual_final
        if epoch < perceptual_warmup_epoch:
            lambda_perceptual = 0.0
        else:
            # Linear ramp over remaining epochs
            progress = (epoch - perceptual_warmup_epoch) / max(1, config.EPOCHS - perceptual_warmup_epoch)
            lambda_perceptual = lambda_perceptual_final * progress
        
        for batch_idx, (masked_img, mask, real_img) in enumerate(loader):
             # SPEED OPTIMIZATION 3: non_blocking=True for async transfers to the GPU
            masked_img = masked_img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            real_img = real_img.to(device, non_blocking=True)

            # Generate the fake repair batch
            gen_img = generator(masked_img, mask)
            
            # The AI only correctly replaces the hole. We compose the final image exactly as infer.py does!
            comp_img = (real_img * mask) + (gen_img * (1 - mask))
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Goal 1: Discriminate the Real Images as True (1.0)
            # Pass mask so discriminator knows WHICH region to scrutinize
            pred_real = discriminator(real_img, mask)
            valid = torch.ones_like(pred_real, device=device)
            loss_real = adversarial_loss(pred_real, valid)

            # Goal 2: Discriminate the Fake composed image as False (0.0)
            fake = torch.zeros_like(pred_real, device=device)
            pred_fake = discriminator(comp_img.detach(), mask)
            loss_fake = adversarial_loss(pred_fake, fake)

            # Combine Discriminator loss
            loss_D = (loss_real + loss_fake) / 2.0
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # The Generator wants the Discriminator to believe its fakes are Real! (True / 1.0)
            pred_fake_G = discriminator(comp_img, mask)
            g_adv_loss = adversarial_loss(pred_fake_G, valid)
            
            # Mathematical structural context loss to make sure the face/eyes/walls line up
            g_pixel_loss = inpainting_loss(gen_img, real_img, mask)
            
            # Semantic Texture Loss using VGG-16!
            g_perceptual_loss = perceptual_criterion(comp_img, real_img)

            # Total Generator Loss combines all three!
            loss_G = g_pixel_loss + (lambda_adv * g_adv_loss) + (lambda_perceptual * g_perceptual_loss)
            loss_G.backward()
            
            # Anti-Explosion gradient clipping to prevent static pixel generation!
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            
            optimizer_G.step()
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

            if batch_idx % 20 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

        avg_loss_G = epoch_loss_G / len(loader)
        avg_loss_D = epoch_loss_D / len(loader)
        logger.info(f"--> Epoch {epoch} completed in {time.time() - start_time:.2f} seconds. Avg G: {avg_loss_G:.4f} | Avg D: {avg_loss_D:.4f}")
        
        # Save the UNet weights
        torch.save(generator.state_dict(), "checkpoints/checkpoint.pth")
        
        # Step the learning rate schedulers
        scheduler_G.step()
        scheduler_D.step()

if __name__ == '__main__':
    main()