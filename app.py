import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
from src.models.unet import UNet
from src.config import IMG_SIZE

# 1. Initialize the Device and Model once globally
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Model on {device}...")

model = UNet().to(device)
if os.path.exists("checkpoints/checkpoint.pth"):
    model.load_state_dict(torch.load("checkpoints/checkpoint.pth", map_location=device))
    model.eval()
    print("State dict loaded successfully!")
else:
    print("WARNING: checkpoints/checkpoint.pth not found! Model will generate noise.")

# Set up the resizing transform based on training resolution
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

def inpaint(dict_input):
    """
    Gradio expects this function when dealing with an ImageEditor (sketch tool).
    dict_input contains:
       'background': The original uploaded PIL Image
       'layers': A list of PIL Images representing the painted custom layers
       'composite': A flattened PIL image of the background + layers combined
    """
    if dict_input is None or 'background' not in dict_input or dict_input['background'] is None:
        return None

    # Load original image
    background_pil = dict_input['background'].convert("RGB")
    
    # Check if user painted anything
    if 'layers' in dict_input and dict_input['layers']:
        # The user's brush strokes are in layers[0]
        # It comes as an RGBA image where brush strokes are opaque and rest is transparent
        layer_pil = dict_input['layers'][0]
        # Extract alpha channel (A) to serve as our binary mask!
        # Values will be 0 (transparent) and 255 (drawn)
        mask_layer_np = np.array(layer_pil)[:, :, 3] 
    else:
        # User didn't draw anything
        return background_pil

    # Convert drawn mask to binary: 0 for drawing (hole), 1 for visible background
    # Our UNet expects: 0 = hole, 1 = context
    # Gradio drawings will have alpha > 0 where user painted
    mask_binary_np = np.where(mask_layer_np > 0, 0.0, 1.0).astype(np.float32)

    # Calculate aspect-ratio preserving dimensions that are multiples of 16
    # This prevents the network from failing on distorted, squashed shapes!
    w, h = background_pil.size
    max_size = 256.0
    scale = max_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    new_h = max(16, (new_h // 16) * 16)
    new_w = max(16, (new_w // 16) * 16)

    # We need to resize BOTH the background and the mask
    resized_bg = background_pil.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    img_tensor = T.ToTensor()(resized_bg).unsqueeze(0).to(device)  # (1, 3, new_h, new_w)
    
    # Mask to tensor (needs resize too - use NEAREST to avoid anti-aliased edge leaks!)
    mask_pil = Image.fromarray((mask_binary_np * 255).astype(np.uint8), mode='L')
    mask_resize_pil = mask_pil.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    mask_tensor = T.ToTensor()(mask_resize_pil).unsqueeze(0).to(device) # (1, 1, new_h, new_w)

    # 3. Create the input exactly as train.py and infer.py do
    masked_img = img_tensor * mask_tensor

    # 4. Forward Pass through the UNet
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device=="cuda")):
            pred = model(masked_img, mask_tensor)
            
    # 5. Composite Final Output
    final_output = (img_tensor * mask_tensor) + (pred * (1 - mask_tensor))

    # Convert back to PIL Image for Gradio
    final_output_cpu = final_output.squeeze(0).cpu().numpy()
    # PyTorch is CHW [C, H, W], PIL expects HWC [H, W, C]
    final_output_cpu = np.transpose(final_output_cpu, (1, 2, 0))
    final_output_cpu = (final_output_cpu * 255).astype(np.uint8)
    
    return Image.fromarray(final_output_cpu)

# 2. Setup the Gradio Web Interface
with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.Markdown("# Image Inpainting AI")
    gr.Markdown("Upload an image, use the brush tool to highlight unwanted objects, and hit **Inpaint** to restore it!")
    
    with gr.Row():
        with gr.Column():
            # Modern Gradio ImageEditor with brush support
            img_input = gr.ImageEditor(
                type="pil",
                image_mode="RGBA",
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"), 
                label="Original Image & Brush Mask"
            )
            inpaint_btn = gr.Button("🎨 Inpaint Photo", variant="primary")
            
        with gr.Column():
            img_output = gr.Image(label="Inpainted Result", type="pil")
            
    # Connect the UI to the function
    inpaint_btn.click(fn=inpaint, inputs=img_input, outputs=img_output)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
