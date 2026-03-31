import torch
import torchvision.transforms as T
from PIL import Image
from src.models.unet import UNet
import os
import glob
from src.utils.visualize import save_sample
import random
from src.config import IMG_SIZE

def infer():
    # 1. Setup the GPU and Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference running on {device}")
    
    model = UNet().to(device)
    if os.path.exists("checkpoints/checkpoint.pth"):
        model.load_state_dict(torch.load("checkpoints/checkpoint.pth", map_location=device))
        print("Loaded checkpoints/checkpoint.pth successfully!")
    else:
        print("ERROR: checkpoints/checkpoint.pth not found! Did you run train.py?")
        return
    
    model.eval()

    # 2. Find any image in data/tests
    images = glob.glob("data/tests/*.*")
    if len(images) == 0:
        print("Could not find any images in data/tests/")
        return
        
    for img_path in images:
        print(f"Testing on {img_path}...")

        # Load and transform image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Could not load image: {e}")
            continue

        transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 3. Create two different masks to test both approaches
        _, _, h, w = img_tensor.shape
        
        # Mode 1: Single large black square
        mask_single = torch.ones((1, 1, h, w)).to(device)
        mask_size_single = random.randint(40, 80)
        x_s = random.randint(0, w - mask_size_single)
        y_s = random.randint(0, h - mask_size_single)
        mask_single[:, :, y_s:y_s+mask_size_single, x_s:x_s+mask_size_single] = 0

        # Mode 2: 5 to 10 medium black squares (covering bad pixels)
        mask_multi = torch.ones((1, 1, h, w)).to(device)
        num_patches = random.randint(5, 10)
        for _ in range(num_patches):
            mask_size = random.randint(15, 30)
            x_m = random.randint(0, w - mask_size)
            y_m = random.randint(0, h - mask_size)
            mask_multi[:, :, y_m:y_m+mask_size, x_m:x_m+mask_size] = 0

        filename = os.path.basename(img_path)

        for mode_name, mask in [('single', mask_single), ('multi', mask_multi)]:
            # The area to be predicted is blacked out
            masked_img = img_tensor * mask

            # 4. Neural Network Inference
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device=="cuda")):
                    pred = model(masked_img, mask)
            
            # 5. Combine the image: Original pixels where mask is 1, Predicted pixels where mask is 0
            final_output = (img_tensor * mask) + (pred * (1 - mask))

            # 6. Save the preview comparison to disk
            output_path = f"outputs/result_{mode_name}_{filename}"
            save_sample(masked_img.cpu(), final_output.cpu(), img_tensor.cpu(), output_path)
            print(f"  -> Saved {mode_name} result to '{output_path}'")

if __name__ == '__main__':
    infer()