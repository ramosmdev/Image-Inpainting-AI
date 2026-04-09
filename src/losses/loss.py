import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# New Adversarial Loss for LSGAN (Least Squares GAN)
# Provides non-saturating gradients and better stability compared to BCE
adversarial_loss = nn.MSELoss()

def inpainting_loss(pred, target, mask):
    missing = 1 - mask
    
    # Calculate the average error ONLY inside the missing hole
    hole_error = torch.abs(pred - target) * missing
    hole_loss = hole_error.sum() / (missing.sum() + 1e-8) # Added 1e-8 to prevent DivByZero NaN!
    
    # Calculate the average error for the rest of the context image!
    valid_error = torch.abs(pred - target) * mask
    valid_loss = valid_error.sum() / (mask.sum() + 1e-8)
    
    # The network must learn to transition context smoothly!
    return (hole_loss * 6.0) + valid_loss

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained VGG-16 without fully connected layers
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # We extract features at specific depth levels of the VGG network
        self.blocks = nn.ModuleList([
            vgg[:4],
            vgg[4:9],
            vgg[9:16]
        ])
        
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # VGG expects images normalized with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        x = (x - mean) / std
        y = (y - mean) / std
        
        loss = 0.0
        fx, fy = x, y
        for block in self.blocks:
            fx = block(fx)
            fy = block(fy)
            # Compare feature maps using L1 distance
            loss += torch.nn.functional.l1_loss(fx, fy)
        return loss