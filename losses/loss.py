import torch

def inpainting_loss(pred, target, mask):
    missing = 1 - mask
    loss = torch.abs(pred - target) * missing
    return loss.mean()