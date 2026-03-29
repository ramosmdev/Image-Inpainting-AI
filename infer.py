import torch
import torchvision.transforms as T
from PIL import Image
from src.models.unet import UNet
import os
import glob
from src.utils.visualize import save_sample
import random

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
        
    img_path = images[0]
    print(f"Testing on {img_path}...")

    # Load and transform image
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Could not load image: {e}")
        return

    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 3. Create a random square mask
    _, _, h, w = img_tensor.shape
    mask = torch.ones((1, 1, h, w)).to(device)
    mask_size = random.randint(20, 50)
    x = random.randint(0, w - mask_size)
    y = random.randint(0, h - mask_size)
    mask[:, :, y:y+mask_size, x:x+mask_size] = 0

    # The area to be predicted is blacked out
    masked_img = img_tensor * mask

    # 4. Neural Network Inference
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device=="cuda")):
            pred = model(masked_img, mask)
    
    # 5. Combine the image: Original pixels where mask is 1, Predicted pixels where mask is 0
    final_output = (img_tensor * mask) + (pred * (1 - mask))

    # 6. Save the preview comparison to disk
    # Grabs the first image from the batch
    save_sample(masked_img.cpu(), final_output.cpu(), img_tensor.cpu(), "outputs/demo_result.png")
    print("Success! Open 'outputs/demo_result.png' in your folder to see your AI inpainting results!")

if __name__ == '__main__':
    infer()