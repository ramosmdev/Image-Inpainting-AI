from datasets.inpainting_dataset import InpaintingDataset
from models.unet import UNet
from losses.loss import inpainting_loss
import config
import torch
from torch.utils.data import DataLoader
import time

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # SPEED OPTIMIZATION 1: Optimizes convolution algorithms if input size doesn't change
        torch.backends.cudnn.benchmark = True 

    dataset = InpaintingDataset("dataset/val2017/")
    # SPEED OPTIMIZATION 2: num_workers loads images in background, pin_memory speeds up CPU->GPU copy
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting training on {device}...")
    for epoch in range(10):
        start_time = time.time()
        for batch_idx, (masked_img, mask, real_img) in enumerate(loader):
             # SPEED OPTIMIZATION 3: non_blocking=True for async transfers to the GPU
            masked_img = masked_img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            real_img = real_img.to(device, non_blocking=True)

            pred = model(masked_img, mask)

            loss = inpainting_loss(pred, real_img, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Limit print statements to avoid terminal lag slowing down the loop!
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

        print(f"--> Epoch {epoch} completed in {time.time() - start_time:.2f} seconds. Final loss: {loss.item():.4f}")
        # SAVE THE MODEL AT THE END OF EACH EPOCH!
        torch.save(model.state_dict(), "checkpoint.pth")

if __name__ == '__main__':
    main()